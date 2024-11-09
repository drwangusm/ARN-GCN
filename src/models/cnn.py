from keras.layers import Dense, Dropout, Flatten
from keras.layers import Reshape

from keras.models import Model
from keras import regularizers

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications import InceptionV3
from keras.applications import ResNet50, ResNet101, ResNet152
from keras.applications import ResNet50V2, ResNet101V2, ResNet152V2

import tensorflow as tf

from models import i3d_inception as i3d

import os
if os.path.exists('/home/USER/.keras/models/'):
    # Hard-coded weights for vgg16 only, not necessary for the other archs
    IMAGENET_FRAMES = '/home/USER/.keras/models/{}_weights_tf_dim_ordering_tf_kernels_notop.h5'
    IMAGENET_FRAMES_TOP = '/home/USER/.keras/models/{}_weights_tf_dim_ordering_tf_kernels.h5'
    
    UCF101_FRAMES = 'models/cnn/cuhk_action_spatial_vgg_16_split1.h5'
    UCF101_FLOWS = 'models/cnn/cuhk_action_temporal_vgg_16_split1.h5'

def manual_load_weights(model, weights_path):
    """ 
        Loading weights from my snapshots.
        That for some reason do not follow the expected Keras format when trained with 2 gpus
    """
    import h5py
    
    with h5py.File(weights_path, 'r') as weights_file:
        if 'model_1' not in weights_file.keys():
            model.load_weights(weights_path)
        else:
            model_w = weights_file['model_1']
            
            for layer_name, layer_grp in model_w.items():
                l = model.get_layer(layer_name)
                layer_weights = [ w[()] for w in layer_grp.values()][::-1]
                l.set_weights(layer_weights)

def get_model(cnn_arch, data_type, output_size=1, drop_rate=0.5, 
        weight_decay=None, initial_weights=None, stack_length=None,
        output_layer=None, load_top=True):
    """
    Build model given input parameters.
    
    In order to get vgg16 model with imagenet weights in top layers also: 
        vgg16, 'frames', load_top = True, initial_weights='imagenet'

    Parameters
    ----------
    cnn_arch : str
        inception-v3, vgg16, vgg19, resnet-xxx, etc.
    data_type : str
        frames or flows.
    output_size : int, optional
        Number of classes for predictions/last layer. The default is 1.
    drop_rate : float, optional
        Dropout rate. The default is 0.5.
    weight_decay : float, optional
        UNTESTED. The default is None.
    initial_weights : TYPE, optional
        Either None, 'imagenet', 'ucf101' or fullpath. The default is None.
    stack_length : int, optional
        How many flows will be stacked. The default is None.
    output_layer : str, optional
        Name of layer to be set as output. The default is None.
        Common choices: 'fc1', 'fc2' and 'predictions' (vgg16)
    load_top : bool, optional
        Whether to load weights for top layers for 'imagenet' or 'ucf101'.
        The default is True.

    Returns
    -------
    model : tensorflow.keras.models.Model
        Generated model.
    """
    
    num_channels = (3 if data_type == 'frames' else 2*stack_length)
    
    if cnn_arch.startswith('vgg'):
        model = build_vgg_model(num_channels, cnn_arch, data_type, 
                output_size=output_size, drop_rate=drop_rate, 
                weight_decay=weight_decay, initial_weights=initial_weights, 
                load_top=load_top)
    elif cnn_arch.lower() == 'inception-v3':
        model = build_inception_model(num_channels, 
                # data_type, 
                output_size=output_size, drop_rate=drop_rate, 
                # weight_decay=weight_decay, 
                initial_weights=initial_weights, 
                load_top=load_top)
    elif cnn_arch.lower().startswith('resnet'):
        model = build_resnet_model(num_channels, cnn_arch, 
                # data_type, 
                output_size=output_size, drop_rate=drop_rate, 
                # weight_decay=weight_decay, 
                initial_weights=initial_weights, 
                load_top=load_top)
    elif cnn_arch.lower() == 'i3d':
        model = build_i3d_model(data_type, stack_length=stack_length,
                output_size=output_size, drop_rate=drop_rate, 
                # weight_decay=weight_decay, 
                initial_weights=initial_weights, 
                load_top=load_top)
        
    if output_layer is not None:
        if not cnn_arch.lower() == 'i3d':
            out_layer = model.get_layer(output_layer).output
        else:
            out_layer = Reshape((1024,), name='squeeze')(
                model.get_layer(output_layer).output)
        
        model_name = '_'.join([cnn_arch, data_type, output_layer])
        model = Model(inputs=model.input,
            outputs=out_layer,
            name=model_name)
    
    return model

def build_inception_model(num_channels, output_size=1000, 
        drop_rate=0.5, initial_weights=None, load_top=True):
    weights = ('imagenet' if initial_weights == 'imagenet' else None)
    classes = (1000 if initial_weights == 'imagenet' else output_size)
    
    model = InceptionV3(include_top=load_top, weights=weights,
                    input_shape=(299, 299, num_channels), classes=classes)
    
    if initial_weights is not None and initial_weights != 'imagenet':
        model.load_weights(initial_weights)
    
    return model

def build_i3d_model(data_type, output_size=400, stack_length=64,
        drop_rate=0.5, initial_weights=None, load_top=True):
    if initial_weights is None or initial_weights not in ['imagenet','kinetics']:
        weights = None
    else:
        weights_prefix = ('rgb_' if data_type == 'frames' else 'flow_')
        weights_sufix = ('imagenet_and_kinetics' if initial_weights == 'imagenet' 
                         else 'kinetics_only')
        weights = weights_prefix + weights_sufix
    
    if data_type == 'frames':
        input_shape = (stack_length, 224, 224, 3)
    elif data_type == 'flows':
        input_shape = (stack_length, 224, 224, 2)
    
    classes = (400 if weights in i3d.WEIGHTS_NAME else output_size)
    # Maybe do not use load_top, set include_top as true always?
    model = i3d.Inception_Inflated3d(include_top=load_top,
                weights=weights,
                # input_tensor=None,
                # input_shape=None, # (NUM_FRAMES, 224, 224, 3)
                input_shape=input_shape,
                dropout_prob=drop_rate,
                endpoint_logit=False, # [True]
                classes=classes)
    # model = i3d.Inception_Inflated3d(include_top=load_top, weights=weights,
    #                 input_shape=(224, 224, num_channels), classes=classes)
    
    if initial_weights is not None and initial_weights not in ['imagenet','kinetics']:
        model.load_weights(initial_weights)
    
    return model

def build_resnet_model(num_channels, cnn_arch, output_size=1000, 
        drop_rate=0.5, initial_weights=None, load_top=True):
    weights = ('imagenet' if initial_weights == 'imagenet' else None)
    classes = (1000 if initial_weights == 'imagenet' else output_size)
    # Maybe do not use load_top, set include_top as true always?
    
    builders = [ResNet50, ResNet101, ResNet152, 
                ResNet50V2, ResNet101V2, ResNet152V2]
    keys = [ f.__name__.lower() for f in builders ]
    dict_builders = dict(zip(keys, builders))
    arch_builder = dict_builders[cnn_arch.lower()]
    
    model = arch_builder(include_top=load_top, weights=weights,
                    input_shape=(224, 224, num_channels), classes=classes)
    
    if initial_weights is not None and initial_weights != 'imagenet':
        model.load_weights(initial_weights)
    
    return model

def build_vgg_model(num_channels, cnn_arch, data_type, output_size=1, 
        drop_rate=0.5, weight_decay=None, initial_weights=None, load_top=True):
    if cnn_arch == 'vgg19':
        cnn_notop = VGG19(weights=None, include_top=False,
            input_shape=(224, 224, num_channels))
    elif cnn_arch == 'vgg16':
        # If weight is not in IMAGENET_FRAMES, need to download first:
        # cnn_notop = VGG16(weights='imagenet', include_top=False,
        cnn_notop = VGG16(weights=None, include_top=False,
            input_shape=(224, 224, num_channels))
    else:
        print("Invalid cnn architecture option:", cnn_arch)
            
    if load_top and initial_weights in ['imagenet', 'ucf101']:
        # Overwrites output_size, so that top can be loaded
        output_size = (1000 if initial_weights=='imagenet' else 101)
    
    kernel_regularizer = (None if weight_decay is None else
        regularizers.l2(weight_decay) )
    
    # Top: Classification block
    x = Flatten(name='flatten')(cnn_notop.output)
    x = Dense(4096, activation='relu', name='fc1', 
        kernel_regularizer=kernel_regularizer)(x)
    x = Dropout(drop_rate)(x)
    x = Dense(4096, activation='relu', name='fc2', 
        kernel_regularizer=kernel_regularizer)(x)
    x = Dropout(drop_rate)(x)
    activation = ('sigmoid' if output_size == 1 else 'softmax')
    x = Dense(output_size, activation=activation, name='predictions', 
        kernel_regularizer=None)(x)
        # kernel_regularizer=kernel_regularizer)(x)
    
    model = Model(inputs=cnn_notop.input, outputs=x)
    
    if initial_weights is not None and initial_weights != 'random':
        if initial_weights == 'imagenet':
            if output_size == 1000:
                weights_path = (IMAGENET_FRAMES_TOP if data_type == 'frames' 
                    else 'dummy_flows_{}').format(cnn_arch)
                model_loader = model
            else:
                weights_path = (IMAGENET_FRAMES if data_type == 'frames' 
                    else IMAGENET_FLOWS).format(cnn_arch)
                model_loader = cnn_notop
        elif initial_weights == 'ucf101':
            if cnn_arch != 'vgg16':
                print("UCF101 pre-trained weights only available for vgg16 architecture")
                return None
            
            weights_path = (UCF101_FRAMES if data_type == 'frames' 
                else UCF101_FLOWS)
            
            if output_size != 101:
                x = Flatten(name='flatten')(cnn_notop.output)
                x = Dense(4096, activation='relu', name='fc1')(x)
                x = Dense(4096, activation='relu', name='fc2')(x)
                x = Dense(101, activation='softmax', name='predictions')(x)
                model_loader = Model(inputs=cnn_notop.input, outputs=x)
            else:
                model_loader = model
        else:
            weights_path = initial_weights
            model_loader = model
        
        if initial_weights == 'imagenet' or initial_weights == 'ucf101':
            model_loader.load_weights(weights_path, by_name=False)
        else:
            # manual_load_weights(model, initial_weights)
            model.load_weights(weights_path)
    
    return model
    