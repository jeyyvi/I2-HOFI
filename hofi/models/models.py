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
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set up feature dimensions based on the base model's output shape
        dims = list(self.base_model.output.shape)[1:]
        self.base_channels = dims[2]
        self.feat_dim = int(self.base_channels) * self.pool_size * self.pool_size

        # Initialize ROIs for pooling
        self.rois_mat = getROIS(
            resolution=self.ROIS_resolution,
            gridSize=self.ROIS_grid_size, 
            minSize=self.minSize
        )
        self.num_rois = self.rois_mat.shape[0]

        # Compute the number of nodes after reshaping the ROI tensor
        self.cnodes = (int(self.base_channels) // self.gcn_outfeat_dim) * self.pool_size * self.pool_size

        # Construct adjacency matrices and required layers
        self._construct_adjacency()
        self._construct_layers()

    def _construct_adjacency(self):
        """Create intra- and inter-ROI adjacency matrices."""
        from spektral.layers import GCNConv
        import scipy.sparse as sp
        
        # Intra-ROI adjacency (cnodes x cnodes)
        A1 = np.ones((self.cnodes, self.cnodes), dtype='float32')
        cfltr1 = GCNConv.preprocess(A1).astype('f4')
        
        # Convert to dense tensor (handle both sparse matrix and numpy array)
        if sp.issparse(cfltr1):
            A_intra = tf.constant(cfltr1.toarray(), dtype=tf.float32)
        else:
            A_intra = tf.constant(cfltr1, dtype=tf.float32)

        # Inter-ROI adjacency (num_rois+1 x num_rois+1)
        A2 = np.ones((self.num_rois + 1, self.num_rois + 1), dtype='float32')
        cfltr2 = GCNConv.preprocess(A2).astype('f4')
        
        # Convert to dense tensor (handle both sparse matrix and numpy array)
        if sp.issparse(cfltr2):
            A_inter = tf.constant(cfltr2.toarray(), dtype=tf.float32)
        else:
            A_inter = tf.constant(cfltr2, dtype=tf.float32)

        # Store adjacency matrices
        self.Adj = [A_intra, A_inter]

    def _temp_nodes_transform(self, roi):
        # Flatten spatial dimensions into a single node dimension for GNN processing
        reshaped_data = tf.reshape(roi, (-1, self.pool_size * self.pool_size, self.base_channels))

        # Split along the channel dimension
        splits = tf.split(reshaped_data, num_or_size_splits=int(self.base_channels) // self.gcn_outfeat_dim, axis=2)

        # Concatenate splits along the node dimension
        joined = tf.concat(splits, 1)

        return joined

    def _extract_roi_nodes(self, x0, base_out):
        # Apply ROI pooling on the backbone output to extract regional features
        roi_pool = self.roi_pooling(x0)

        jcvs = []
        # Loop through each ROI and process it individually
        for j in range(self.num_rois):
            roi_crop = crop(1, j, j + 1)(roi_pool)
            lname = 'roi_lambda_' + str(j)
            x = layers.Lambda(squeezefunc, name=lname)(roi_crop)
            x = layers.Reshape((self.feat_dim,))(x)
            jcvs.append(x)

        # Resize backbone output to match ROI pool size
        if self.pool_size != base_out.shape[1]: 
            base_out = layers.Lambda(lambda x: tf.image.resize(x, size=(self.pool_size, self.pool_size)), name='Lambda_img_2')(base_out)

        x = layers.Reshape((self.feat_dim,))(base_out)
        jcvs.append(x)
        
        jcvs = layers.Lambda(stackfunc, name='lambda_stack')(jcvs)
        jcvs = self.roi_droput_1(jcvs)

        return jcvs

    def _construct_layers(self):
        # Upsample base network output for finer ROI extraction
        self.upsampling_layer = layers.Lambda(
            lambda x: tf.image.resize(x, size=(self.ROIS_resolution, self.ROIS_resolution)), 
            name='UpSample'
        )

        # ROI pooling layer
        self.roi_pooling = RoiPoolingConv(
            pool_size=self.pool_size, 
            num_rois=self.num_rois, 
            rois_mat=self.rois_mat
        )

        # Dropout layers
        self.roi_droput_1 = tf.keras.layers.Dropout(self.dropout_rate, name='DOUT_1')
        self.roi_droput_2 = tf.keras.layers.Dropout(self.dropout_rate, name='DOUT_2')

        # TimeDistributed layers
        self.timedist_layer1 = layers.TimeDistributed(
            layers.Reshape((self.pool_size, self.pool_size, self.base_channels)), 
            name='TD_Layer1'
        )
        
        self.timedist_layer2 = layers.TimeDistributed(
            layers.Lambda(self._temp_nodes_transform), 
            name='TD_Layer2'
        )

        # GNN layers
        if self.gnn1_layr:
            self.tgcn_1 = APPNPConv(
                self.gcn_outfeat_dim, 
                alpha=self.alpha, 
                propagations=1, 
                mlp_activation=self.appnp_activation, 
                use_bias=True, 
                name='GNN_1'
            )

        if self.gnn2_layr:
            self.tgcn_2 = GATConv(
                self.gat_outfeat_dim,
                attn_heads=self.attn_heads,
                concat_heads=self.concat_heads,
                dropout_rate=self.dropout_rate,
                activation=self.gat_activation,
                kernel_regularizer=l2(self.l2_reg),
                attn_kernel_regularizer=l2(self.l2_reg),
                bias_regularizer=l2(self.l2_reg), 
                name='GNN_2'
            )

        # Final layers
        self.GlobAttpool = GlobalAttentionPool(self.gcn_outfeat_dim * 2, name='GlobalAttnPool')
        self.BN2 = layers.BatchNormalization(name='BN')
        self.Dense = layers.Dense(self.nb_classes, activation='softmax', name='Fully_Conn')

    def call(self, inputs, training=None):
        # Get feature maps from base model
        base_out = self.base_model(inputs, training=training)
        if self.track_feat:  
            self.base_out = tf.identity(base_out)
        
        # Upsample feature maps
        x0 = self.upsampling_layer(base_out)

        # Extract and process ROIs
        rois = self._extract_roi_nodes(x0, base_out)

        # Apply time-distributed layers
        x1 = self.timedist_layer1(rois)
        x1 = self.timedist_layer2(x1)

        # Intra-ROI GNN processing
        splits = tf.split(x1, num_or_size_splits=self.num_rois + 1, axis=1)
        xcoll = []
        for x in splits:
            x = tf.squeeze(x, axis=1)
            if self.gnn1_layr:
                # Call APPNP layer without training parameter
                temp = self.tgcn_1.call([x, self.Adj[0]])
                x = temp + x
            if self.gnn2_layr:
                # Call GAT layer without training parameter - returns (output, attention_weights)
                result = self.tgcn_2.call([x, self.Adj[0]])
                if isinstance(result, tuple):
                    temp = result[0]
                else:
                    temp = result
                temp = temp + x
            xcoll.append(temp)

        x2_intra = tf.concat(xcoll, axis=1)
        
        if self.track_feat:
            self.x2_intra = tf.identity(x2_intra)

        # Inter-ROI GNN processing
        x1 = tf.transpose(x1, perm=[0, 2, 1, 3])
        splits = tf.split(x1, num_or_size_splits=self.cnodes, axis=1)
        xcoll = []
        for x in splits:
            x = tf.squeeze(x, axis=1)
            if self.gnn1_layr:
                # Call APPNP layer without training parameter
                temp = self.tgcn_1.call([x, self.Adj[1]])
                x = temp + x
            if self.gnn2_layr:
                # Call GAT layer without training parameter - returns (output, attention_weights)
                result = self.tgcn_2.call([x, self.Adj[1]])
                if isinstance(result, tuple):
                    temp = result[0]
                else:
                    temp = result
                temp = temp + x
            xcoll.append(temp)

        x2_inter = tf.concat(xcoll, axis=1)
        
        if self.track_feat:
            self.x2_inter = tf.identity(x2_inter)

        # Combine and process
        x2 = tf.concat([x2_intra, x2_inter], axis=1)
        x3 = self.roi_droput_2(x2, training=training)

        # Global attention pooling
        xf = self.GlobAttpool(x3)
        
        if self.track_feat:
            self.GlobAttpool_feat = tf.identity(xf)

        # Final layers
        xf = self.BN2(xf, training=training)
        feat = self.Dense(xf)

        return feat