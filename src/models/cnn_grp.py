from keras import layers
from keras.models import Model

import tensorflow as tf

# import cnn
from models import cnn

from keras.layers import Dropout, Dense  # Import Dropout and Dense layers

def get_model(num_objs, output_size=1, indiv_out=False, indiv_output_size=9, 
              **cnn_kwargs):
    cnn_arch = cnn_kwargs['cnn_arch']
    default_output_layers = dict(vgg16='fc2', i3d='global_avg_pool')
    default_ol = default_output_layers.get(cnn_arch, 'avg_pool')
    
    cnn_kwargs.setdefault('output_layer', default_ol)
    output_layer = cnn_kwargs['output_layer']
    drop_rate = cnn_kwargs.get('drop_rate', 0.5)
    
    cnn_model = cnn.get_model(**cnn_kwargs)
    object_shape = tuple(cnn_model.input.shape[1:])
    
    inputs = create_input(num_objs, object_shape)
    
    cnn_outputs = []
    # for obj_idx in range(num_objs):
    for obj_input in inputs:
        cnn_output = cnn_model(obj_input)
        cnn_outputs.append(cnn_output)
    
    grp_cnn_outs = cnn_outputs.copy()
    ### Creating Group top and predictions
    if output_layer != 'fc2' and cnn_arch == 'vgg16':
        if output_layer == 'flatten':
            fc1_layer = layers.Dense(4096, activation='relu', name='fc1')
            fc1_dr_l = layers.Dropout(drop_rate)
            grp_cnn_outs = [ fc1_dr_l(fc1_layer(o)) for o in grp_cnn_outs ]
        
        fc2_layer = layers.Dense(4096, activation='relu', name='fc2')
        
        grp_cnn_outs = [ fc2_layer(o) for o in grp_cnn_outs ]
        # fc2_dr_l = layers.Dropout(drop_rate)
        # grp_cnn_outs = [ fc2_dr_l(fc2_layer(o)) for o in grp_cnn_outs ]
    
    mode = 'max'
    name_suffix = ''
    pool_out = layers.Maximum(name=mode+name_suffix)(grp_cnn_outs)
    
    if drop_rate > 0:
        pool_out = layers.Dropout(drop_rate, name='dropout_grp')(pool_out)
    
    pred_out = layers.Dense(output_size, activation='softmax', 
                            name='softmax')(pool_out)
    
    ### Creating indivs top and predictions
    if indiv_out:
        ind_cnn_outs = cnn_outputs.copy()
        if output_layer != 'fc2' and cnn_arch == 'vgg16':
            if output_layer == 'flatten':
                fc1_layer = layers.Dense(4096, activation='relu',
                                         name='fc1_indivs')
                fc1_dr_l = layers.Dropout(drop_rate)
                ind_cnn_outs = [ fc1_dr_l(fc1_layer(o)) for o in ind_cnn_outs ]
            
            fc2_layer = layers.Dense(4096, activation='relu', name='fc2_indivs')
            ind_cnn_outs = [ fc2_layer(o) for o in ind_cnn_outs ]
            
        preds_indivs_dr = layers.Dropout(drop_rate, name='dropout_indivs')
        preds_indivs = layers.Dense(indiv_output_size, activation='softmax', 
                            name='softmax_indivs')
        
        preds_indivs_out = []
        for cnn_out in ind_cnn_outs:
            if drop_rate > 0:
                preds_indivs_out.append(preds_indivs(preds_indivs_dr(cnn_out)))
            else:
                preds_indivs_out.append(preds_indivs(cnn_out))
        
        pred_out = [pred_out] + preds_indivs_out
    
    model = Model(inputs=inputs, outputs=pred_out, name="cnn_grp")
    
    return model

def create_input(num_objs, object_shape, num_frames=1, num_extra_objs=0):
    persons = []
    for ind_idx in range(num_objs):
        if num_frames == 1:
            frm = layers.Input(shape=object_shape,
                        name="p{:0>2}".format(ind_idx+1))
                        # name="p{:0>2}_frm{:0>2}".format(ind_idx+1, 1))
            persons.append(frm)
        else:
            person_joints = []
            for frm_idx in range(num_frames):
                frm = layers.Input(shape=object_shape,
                        name="p{:0>2}_frm{:0>2}".format(ind_idx+1, frm_idx+1))
                person_joints.append(frm)
            persons.append(person_joints)
    ret_val = persons
    
    if num_extra_objs > 0:
        extra_objs = []
        for x_obj_idx in range(num_extra_objs):
            x_obj = []
            for frm_idx in range(num_frames):
                frm = layers.Input(shape=object_shape,
                            name="o{}_frm{}".format(x_obj_idx+1, frm_idx+1))
                x_obj.append(frm)
            extra_objs.append(x_obj)
        ret_val = (persons, extra_objs)
    
    return ret_val

def create_top(input_top, drop_rate=0, fc_units=[500,100,100], 
        fc_drop=False):
    x = Dropout(drop_rate)(input_top)
    
    x = Dense(fc_units[0], activation='relu', kernel_initializer=kernel_init, 
        name="fc1")(x)
    if fc_drop: x = Dropout(drop_rate)(x)
    x = Dense(fc_units[1], activation='relu', kernel_initializer=kernel_init, 
        name="fc2")(x)
    if fc_drop: x = Dropout(drop_rate)(x)
    x = Dense(fc_units[2], activation='relu', kernel_initializer=kernel_init, 
        name="f_phi_fc3")(x)
    
    return x