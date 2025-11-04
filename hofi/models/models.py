# Importing tensorflow pacakages
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2

# Importing necessary packages from spektral
from spektral.utils.sparse import sp_matrix_to_sp_tensor 
from spektral.layers import GCNConv, APPNPConv, GATConv, GlobalAttentionPool
from spektral.utils import normalized_adjacency

# from user-defined scripts
from utils import RoiPoolingConv, getROIS, getIntegralROIS, crop, squeezefunc, stackfunc


class Params(Model):
    """
    Params: Initializes parameters for CNN and GNN models.

    This base class centralizes parameter setup for models that combine Convolutional Neural Networks (CNNs) 
    and Graph Neural Networks (GNNs), supporting flexible configuration of graph and image processing layers.

    Parameters:
    - CNN and GNN Config: Includes parameters for CNN backbones, graph convolution layers, attention heads, 
      pooling sizes, dropout rates, and regularization.
    - Backbone Setup: Optionally loads a specified CNN backbone from `tf.keras.applications` and supports 
      freezing the backbone layers for transfer learning.
    - Graph Settings: Configures graph attention and propagation layers with custom dimensions and activation.

    Usage:
    This base class can be inherited by model classes requiring consistent parameter configuration for mixed 
    CNN-GNN architectures.

    """
    def __init__(
        self,
        pool_size=None,
        ROIS_resolution=None,
        ROIS_grid_size=None,
        minSize=None,
        alpha=None,
        nb_classes=None,
        batch_size=None,
        input_sh = (224, 224, 3),
        gcn_outfeat_dim = 256,
        gat_outfeat_dim = 256,
        dropout_rate = 0.2,
        l2_reg = 2.5e-4,
        attn_heads = 1,
        appnp_activation = 'sigmoid',
        gat_activation = 'elu',
        concat_heads = True,
        backbone = None,
        freeze_backbone = None,
        gnn1_layr = True,
        gnn2_layr = True,
        track_feat = False,
        *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.pool_size = pool_size
        self.ROIS_resolution = ROIS_resolution
        self.ROIS_grid_size = ROIS_grid_size
        self.minSize = minSize
        self.alpha = alpha
        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.input_sh = input_sh
        self.gcn_outfeat_dim = gcn_outfeat_dim
        self.gat_outfeat_dim = gat_outfeat_dim
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg  # L2 regularization rate
        self.attn_heads = attn_heads
        self.appnp_activation = appnp_activation
        self.gat_activation = gat_activation
        self.concat_heads = concat_heads
        self.gnn1_layr = gnn1_layr
        self.gnn2_layr = gnn2_layr
        self.track_feat = track_feat
        
        # Load the specified CNN backbone
        base_model_class = getattr(tf.keras.applications, backbone)
        self.base_model = base_model_class(
            weights="imagenet",
            input_tensor = layers.Input( shape = self.input_sh ), 
            include_top = False,
        )      
        
        # Optionally freeze backbone layers
        if freeze_backbone:
            for layer in self.base_model.layers:
                layer.trainable = False           

        # Adjust GAT output node dimension; if concat_heads is True, split gat_outfeat_dim across heads: gat_outfeat_dim = GAT_node_dim * heads 
        if self.concat_heads:    
            self.gat_outfeat_dim = self.gat_outfeat_dim // self.attn_heads
    

class GATConv(GATConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        '''
        Modified _call_dense for compatibility recent TF package
        '''

    def _call_dense(self, x, a):
        shape = tf.shape(a)[:-1]
        if self.add_self_loops:
            a = tf.linalg.set_diag(a, tf.ones(shape, a.dtype))
        x = tf.einsum("...NI , IHO -> ...NHO", x, self.kernel)
        attn_for_self = tf.einsum("...NHI , IHO -> ...NHO", x, self.attn_kernel_self)
        attn_for_neighs = tf.einsum(
            "...NHI , IHO -> ...NHO", x, self.attn_kernel_neighs
        )
        attn_for_neighs = tf.einsum("...ABC -> ...CBA", attn_for_neighs)

        attn_coef = attn_for_self + attn_for_neighs
        attn_coef = tf.nn.leaky_relu(attn_coef, alpha=0.2)

        mask = tf.where(K.eval(a) == 0.0, -10e9, 0.0)
        mask = tf.cast(mask, dtype=attn_coef.dtype)

        attn_coef += mask[..., None, :]
        attn_coef = tf.nn.softmax(attn_coef, axis=-1)
        attn_coef_drop = self.dropout(attn_coef)

        output = tf.einsum("...NHM , ...MHI -> ...NHI", attn_coef_drop, x)

        return output, attn_coef        


# ################################################################################ #
# ############################# Model Definations ################################ #
# ################################################################################ #

class BASE_CNN(Params):
    """
    BASE_CNN: A simple CNN model with a classification head.

    This model uses a backbone CNN followed by a Global Average Pooling (GAP) layer and a dense 
    softmax layer for classification. It is suitable for straightforward image classification tasks 
    and can serve as a baseline for comparison.

    Components:
    - GAP Layer: Aggregates spatial features into a global vector.
    - Dense Layer: Softmax layer for multi-class classification.

    Methods:
    - _construct_layers: Initializes GAP and classification layers.
    - call: Performs forward pass; optionally tracks intermediate features.

    """
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Build required top layers
        self._construct_layers()
        
    def _construct_layers(self):
        # Add a custom classification head
        self.GAP_layer = layers.GlobalAveragePooling2D()
        self.dense = layers.Dense(self.nb_classes, activation="softmax")  #  Final dense layer with softmax activation
        
    def call(self, inputs):
        base_out = self.base_model(inputs)
        x_gap = self.GAP_layer(base_out)
        x = self.dense(x_gap)
        
        # Track features for t-SNE computation
        if self.track_feat:
            self.base_out = tf.identity(base_out)
            self.GlobAttpool_feat = tf.identity(x_gap)
        
        return x


class I2HOFI(Params):
    """
    I2HOFI : (paper: Interweaving Insights: High-Order Feature Interaction for Fine-Grained Visual Recognition)

    I2HOFI is a hybrid CNN-GNN model designed for fine-grained visual recognition. It uses Graph Neural 
    Networks (GNNs) to capture complex interactions both within individual regions (intra-ROI) and 
    across multiple regions of interest (inter-ROI) in an image. This model combines APPNP and GAT 
    layers for high-order feature interaction, enabling robust, context-aware representations.

    Key Components:
    - ROI Pooling: Extracts features from specified regions, with the full image as an additional ROI.
    - Intra- and Inter-ROI GNNs: APPNP and GAT layers capture intra- and cross-region dependencies.
    - Global Attention Pooling: Combines ROI features into a final representation for classification.

    Methods:
    - __init__: Initializes the model, sets feature dimensions, and configures ROI information.
    - _construct_adjacency: Builds adjacency matrices for intra- and inter-ROI GNN layers.
    - _extract_roi_nodes: Pools and reshapes each ROI for processing.
    - _temp_nodes_transform: Reshapes and segments ROI tensors for graph convolution.
    - _construct_layers: Defines all model layers, including backbone CNN, GNN layers, and attention pooling.
    - call: Forward pass that applies feature extraction, ROI pooling, graph convolutions, and attention pooling for final predictions.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set up feature dimensions based on the base model's output shape
        dims = list(self.base_model.output.shape)[1:]
        self.base_channels = dims[2]
        self.feat_dim = int(self.base_channels) * self.pool_size * self.pool_size

        # Initialize ROIs for pooling
        self.rois_mat =  getROIS(
            resolution = self.ROIS_resolution,
            gridSize = self.ROIS_grid_size, 
            minSize = self.minSize
            )
        self.num_rois = self.rois_mat.shape[0]

        # Compute the number of nodes after reshaping the ROI tensor, e.g., for a 3 x 3 x 2048 tensor, 
        # the total nodes are 3 * 3 * (2048 // 512) = 36, where 512 is the target output dimension per node for the GNN
        self.cnodes = (int(self.base_channels) // self.gcn_outfeat_dim) * self.pool_size * self.pool_size

        # Construct adjacency matrices and required layers
        self._construct_adjecency()
        self._construct_layers()


    def _construct_adjecency(self):
        # Adjacency matrix for intra-ROI processing; if there are 36 nodes, then A_intra is 36 x 36
        A1 = np.ones((self.cnodes, self.cnodes), dtype = 'int') 
        cfltr1 = GCNConv.preprocess(A1).astype('f4')   # Normalize Adjacency matrix
        A_intra = layers.Input(tensor=sp_matrix_to_sp_tensor(cfltr1), name = 'AdjacencyMatrix1') 

        # Adjacency matrix for inter-ROI processing; if there are 26 ROIs, then A_inter is 26 x 26
        A2 = np.ones((self.num_rois + 1, self.num_rois + 1), dtype = 'int') 
        cfltr2 = GCNConv.preprocess(A2).astype('f4')  # Normalize Adjacency matrix
        A_inter = layers.Input(tensor=sp_matrix_to_sp_tensor(cfltr2), name = 'AdjacencyMatrix2') 

        # Combine inter- and intra-ROI adjacency matrices into a list
        self.Adj = [A_intra, A_inter]


    def _temp_nodes_transform(self, roi):
        # Flatten spatial dimensions into a single node dimension for GNN processing, e.g., r x 3 x 3 x 2048 --> r x 9 x 2048
        reshaped_data = tf.reshape(roi, (-1, self.pool_size * self.pool_size, self.base_channels))

        # Split along the channel dimension (say 4), e.g., r x 9 x 2048 --> 4 splits of [r x 9 x 512]
        splits = tf.split(reshaped_data, num_or_size_splits=int(self.base_channels) // self.gcn_outfeat_dim, axis=2)

        # Concatenate splits along the node dimension, e.g., 4 splits of [r x 9 x 512] --> r x 36 x 512
        joined = tf.concat(splits, 1)

        return joined


    def _extract_roi_nodes(self, x0, base_out):
        # Apply ROI pooling on the backbone output to extract regional features
        roi_pool = self.roi_pooling(x0)

        jcvs = []
        # Loop through each ROI and process it individually
        for j in range(self.num_rois):
            # Crop each ROI from the pooled features
            roi_crop = crop(1, j, j + 1)(roi_pool)
            lname = 'roi_lambda_' + str(j)

            # Squeeze out unnecessary dimensions and reshape each ROI into a feature vector
            x = layers.Lambda(squeezefunc, name = lname)(roi_crop)
            x = layers.Reshape((self.feat_dim,))(x)

            # Append processed ROI feature to the list
            jcvs.append(x)

        # Resize backbone output to match ROI pool size and treat as an additional ROI
        if self.pool_size != base_out.shape[1]: 
            base_out = layers.Lambda(lambda x: tf.image.resize(x, size = (self.pool_size, self.pool_size)), name = 'Lambda_img_2')(base_out)

        # Reshape and append the resized base output to the list
        x = layers.Reshape((self.feat_dim,))(base_out) # append the original ones
        jcvs.append(x)
        
        # Stack all ROI and base features along a new dimension
        jcvs = layers.Lambda(stackfunc, name = 'lambda_stack')(jcvs)

        # Apply dropout to the stacked ROI features
        jcvs = self.roi_droput_1(jcvs)

        return jcvs


    def _construct_layers(self):
        # Upsample base network output for finer ROI extraction, e.g., Xception [7 x 7 x 2048] -> [42 x 42 x 2048]
        self.upsampling_layer = layers.Lambda(lambda x: tf.image.resize(x, size = (self.ROIS_resolution, self.ROIS_resolution)), name = 'UpSample')

        # ROI pooling layer for extracting features from specified ROIs
        self.roi_pooling = RoiPoolingConv(pool_size = self.pool_size, num_rois = self.num_rois, rois_mat = self.rois_mat)

        # Dropout layer applied after ROI pooling
        self.roi_droput_1 = tf.keras.layers.Dropout(self.dropout_rate, name='DOUT_1')

        # TimeDistributed layer to reshape each ROI for GNN processing; [r x ppC] --> [r x p x p x C]
        # where r: regions, p: pool_size, C: base_channel dimension
        self.timedist_layer1 = layers.TimeDistributed(
            layers.Reshape((self.pool_size, self.pool_size, self.base_channels)), name='TD_Layer1'
        )

        # TimeDistributed layer to transform ROIs to a format compatible with GNN layers
        # Example: ROIs of [r x p x p x C] split into s times [r x p x p x ndim] --> [r x pps x ndim]
        # where ndim: output node dimension, s = C/ndim
        self.timedist_layer2 = layers.TimeDistributed(
            layers.Lambda(self._temp_nodes_transform), name='TD_Layer2'
        )

        if self.gnn1_layr:
            # First GNN layer using APPNP
            self.tgcn_1 = APPNPConv(
                self.gcn_outfeat_dim, 
                alpha = self.alpha, 
                propagations = 1, 
                mlp_activation = self.appnp_activation, 
                use_bias = True, 
                name = 'GNN_1'
            )

        if self.gnn2_layr:
            # Second GNN layer using GAT
            self.tgcn_2 = GATConv(
                self.gat_outfeat_dim,
                attn_heads = self.attn_heads,
                concat_heads = self.concat_heads,
                dropout_rate = self.dropout_rate,
                activation = self.gat_activation,
                kernel_regularizer = l2(self.l2_reg),
                attn_kernel_regularizer = l2(self.l2_reg),
                bias_regularizer = l2(self.l2_reg), 
                name = 'GNN_2'
                )

        # Dropout layer applied after combining all intra- and inter-ROI nodes     
        self.roi_droput_2 = tf.keras.layers.Dropout(self.dropout_rate, name='DOUT_2')

        # Final layers: global attention pooling, batch normalization, and dense layer for classification
        self.GlobAttpool = GlobalAttentionPool(self.gcn_outfeat_dim * 2, name = 'GlobalAttnPool')
        self.BN2 = layers.BatchNormalization(name = 'BN')
        self.Dense = layers.Dense(self.nb_classes, activation='softmax', name='Fully_Conn')

        
    def call(self, inputs):
        # Get feature maps from base model and Track base features if  enabled
        base_out = self.base_model(inputs)
        if self.track_feat:  
            self.base_out = tf.identity(base_out)
        
        # Upsample feature maps
        x0 = self.upsampling_layer(base_out)

        # Extract and process ROIs from upsampled feature maps
        rois = self._extract_roi_nodes(x0, base_out)

        # Apply time-distributed layers to reshape each ROI for GNN processing
        x1 = self.timedist_layer1(rois)
        x1 = self.timedist_layer2(x1)

        # Intra-ROI GNN processing
        splits = tf.split(x1, num_or_size_splits = self.num_rois + 1, axis = 1)
        xcoll = []
        for x in splits:
            x = tf.squeeze(x, axis=1)
            if self.gnn1_layr:
                temp = self.tgcn_1([x, self.Adj[0] ])
                x = temp + x       # Apply residual connection
            if self.gnn2_layr:
                temp = self.tgcn_2([x, self.Adj[0] ])
                temp = temp + x    # Apply residual connection
            xcoll.append(temp)

        # Concatenate results from intra-ROI processing
        x2_intra = tf.concat(xcoll, axis=1)
        
        # Track intra-ROI features if feature tracking is enabled
        if self.track_feat:
            self.x2_intra = tf.identity(x2_intra)

        # Inter-ROI GNN processing
        x1 = tf.transpose(x1, perm=[0, 2, 1, 3]) # Swap dimensions for inter-ROI processing
        splits = tf.split(x1, num_or_size_splits = self.cnodes, axis = 1)
        xcoll = []
        for x in splits:
            x = tf.squeeze(x, axis=1)
            if self.gnn1_layr:
                temp = self.tgcn_1([x, self.Adj[1] ])
                x = temp + x       # Apply residual connection 
            if self.gnn2_layr:
                temp = self.tgcn_2([x, self.Adj[1] ])
                temp = temp + x    # Apply residual connection
            xcoll.append(temp)

        # Concatenate results from inter-ROI processing
        x2_inter = tf.concat(xcoll, axis=1)
        
        # Track inter-ROI features if feature tracking is enabled
        if self.track_feat:
            self.x2_inter = tf.identity(x2_inter)

        # Combine intra- and inter-ROI features and apply dropout
        x2 = tf.concat([x2_intra, x2_inter], axis=1)
        x3 = self.roi_droput_2(x2)

        # Apply global attention pooling
        xf = self.GlobAttpool(x3)
        
        # Track pooled features if feature tracking is enabled
        if self.track_feat:
            self.GlobAttpool_feat = tf.identity(xf)

        # Final batch normalization and dense layer for classification
        xf = self.BN2(xf)
        feat = self.Dense(xf)

        return feat
