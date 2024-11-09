from keras.layers import (Dense, Dropout, Concatenate, Input,
    Add, Maximum, Average, Lambda, Attention, AdditiveAttention, 
    GlobalAveragePooling1D, SpatialDropout1D)
from keras.models import Model

from keras import initializers

from keras import backend as K

import tensorflow as tf

# Hard-coded adjacency matrices for inter-graph

G = [
     # 0) Fully connected (132 total) - Same as inter-direct
    [[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]],
     # 1) Densely connected (78 total)
    [[0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], 
     [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], 
     [1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0], 
     [1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0], 
     [1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0], 
     [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 
     [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1], 
     [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1], 
     [1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1], 
     [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1], 
     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1], 
     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]],
    # 2) Sparsely connected (50 total)
    [[0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], 
     [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0], 
     [0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0], 
     [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
     [1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0], 
     [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
     [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], 
     [1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0], 
     [0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0], 
     [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0], 
     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1], 
     [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0]],
    # 3) Sparsely++ connected (42 total)
   [[0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], 
     [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0], 
     [0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0], 
     [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
     [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0], 
     [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
     [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], 
     [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0], 
     [0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0], 
     [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0], 
     [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1], 
     [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0]] 
   ]

def get_relevant_kwargs(kwargs, func):
    from inspect import signature
    keywords = signature(func).parameters.keys()
    new_kwargs = {}
    for key in keywords:
        if key in kwargs.keys():
            new_kwargs[key] = kwargs[key]
    
    return new_kwargs

def remove_None_kwargs(kwargs):
    new_kwargs = {}
    for key, value in kwargs.items():
        if value is not None:
            new_kwargs[key] = value
    
    return new_kwargs

def get_kernel_init(type, param=None, seed=None):
    kernel_init = None
    if type == 'glorot_uniform':
        kernel_init= initializers.glorot_uniform(seed=seed)
    elif type == 'VarianceScaling':
        kernel_init= initializers.VarianceScaling(seed=seed)
    elif type == 'RandomNormal':
        if param is None:
            param = 0.04
        kernel_init= initializers.RandomNormal(mean=0.0,stddev=param,seed=seed)
    elif type == 'TruncatedNormal':
        if param is None:
            param = 0.045 # Best for non-normalized coordinates
            # param = 0.09 # "Best" for normalized coordinates
        kernel_init=initializers.TruncatedNormal(mean=0.0,stddev=param,seed=seed)
    elif type == 'RandomUniform':
        if param is None:
            param = 0.055 # Best for non-normalized coordinates
            # param = ?? # "Best" for normalized coordinates
        kernel_init= initializers.RandomUniform(minval=-param, maxval=param,
                                                seed=seed)
        
    return kernel_init

def get_model(num_objs, object_shape, rel_type, output_size, 
        kernel_init_type='TruncatedNormal', kernel_init_param=0.045, 
        kernel_init_seed=None, indiv_out=False, indiv_output_size=0, 
        grp_out=True, **f_and_g_kwargs):
    kernel_init = get_kernel_init(kernel_init_type, param=kernel_init_param, 
        seed=kernel_init_seed)
    
    f_phi_model = f_phi(num_objs, object_shape, rel_type, 
        kernel_init=kernel_init, indiv_out=indiv_out, **f_and_g_kwargs)
    
    if not indiv_out:
        top_out = f_phi_model.output
    else:
        top_out, *indivs_top_out = f_phi_model.output
        
        preds_indivs = Dense(indiv_output_size, activation='softmax', 
            kernel_initializer=kernel_init, name='softmax_indivs')
        
        preds_indivs_out = []
        for indiv_top_out in indivs_top_out:
            preds_indivs_out.append(preds_indivs(indiv_top_out))
        
    out_rn = Dense(output_size, activation='softmax', 
        kernel_initializer=kernel_init, name='softmax')(top_out)
    
    if indiv_out:
        if grp_out:
            out_rn = [out_rn] + preds_indivs_out
        else:
            out_rn = preds_indivs_out
    
    model = Model(inputs=f_phi_model.input, outputs=out_rn, name="rel_net")
    
    return model

def make_connections(rel_type, g_theta_model, p1_joints, p2_joints, 
                   persons=[], extra_objs=[], name_suffix='', skip_pool=False):
    g_theta_outs = []
    
    if rel_type == 'inter' or rel_type == 'p1_p2_all_bidirectional':
        ### inter
        # All joints from person1 connected to all joints of person2, and back
        if persons == []:
            for object_i in p1_joints:
                for object_j in p2_joints:
                    g_theta_outs.append(g_theta_model([object_i, object_j]))
            for object_i in p2_joints:
                for object_j in p1_joints:
                    g_theta_outs.append(g_theta_model([object_i, object_j]))
            # rel_out = Average()(g_theta_outs)
            rel_out = pool_relations(g_theta_outs, 'avg', 
                                     name_suffix=name_suffix)
        else:
            inters_out = []
            for p1_idx, p1_joints in enumerate(persons[:-1]):
                for idx, p2_joints in enumerate(persons[p1_idx+1:]):
                    p2_idx = idx + p1_idx + 1
                    rels_name = rel_type+"_p{:0>2}-p{:0>2}".format(p1_idx,p2_idx)
                    inter_avg = make_connections(rel_type, g_theta_model, 
                                p1_joints, p2_joints, name_suffix=rels_name)
                    inters_out.append(inter_avg)
            # rel_out = Average()(inters_out)
            rel_out = inters_out
    elif rel_type == 'inter-direct':
        ### inter directional, single avg per player, with all joints pairs
        inters_out = []
        for p1_idx, p1_joints in enumerate(persons):
            p_inters_out = []
            for p2_idx, p2_joints in enumerate(persons):
                if p1_idx != p2_idx:
                    rels_name = rel_type+"_p{:0>2}-p{:0>2}".format(p1_idx,p2_idx)
                    inter_rels = make_connections('p1_p2_all', g_theta_model, 
                                   p1_joints, p2_joints, name_suffix=rels_name,
                                   skip_pool=True)
                    p_inters_out += inter_rels
            rels_name = rel_type+"_p{:0>2}".format(p1_idx)
            p_inters_avg = pool_relations(p_inters_out, 'avg', 
                                          name_suffix=rels_name)
            inters_out.append(p_inters_avg)
        rel_out = inters_out
    # elif rel_type == 'inter-direct2':
    elif rel_type in ['inter-direct2','inter-direct-avg']:
        ### inter directional, pre-avg per player connection: px-py
        inters_out = []
        for p1_idx, p1_joints in enumerate(persons):
            p_inters_out = []
            for p2_idx, p2_joints in enumerate(persons):
                if p1_idx != p2_idx:
                    rels_name = rel_type+"_p{:0>2}-p{:0>2}".format(p1_idx+1,p2_idx+1)
                    inter_avg = make_connections('p1_p2_all', g_theta_model, 
                                   p1_joints, p2_joints, name_suffix=rels_name)
                    p_inters_out.append(inter_avg)
            rels_name = rel_type+"_p{:0>2}".format(p1_idx)
            p_inters_avg = pool_relations(p_inters_out, 'avg', 
                                          name_suffix=rels_name)
            inters_out.append(p_inters_avg)
        rel_out = inters_out
    elif rel_type == 'inter-direct-att':
        #### [inter-direct-att]
        ## inter directional, pre-avg per player connection, then att pool
        inters_out = []
        for p1_idx, p1_joints in enumerate(persons):
            p_inters_out = []
            for p2_idx, p2_joints in enumerate(persons):
                if p1_idx != p2_idx:
                    rels_name = rel_type+"_p{:0>2}-p{:0>2}".format(p1_idx+1,p2_idx+1)
                    inter_avg = make_connections('p1_p2_all', g_theta_model, 
                                   p1_joints, p2_joints, name_suffix=rels_name)
                    p_inters_out.append(inter_avg)
            inters_out.append(p_inters_out)
        inters_att_out = pool_relations(inters_out, 'att-luong_inter', 
                                      name_suffix=rel_type)
        rel_out = inters_att_out
    elif rel_type == 'inter-mixed':
        ### inter bi-directional players, unidirectional joints
        inters_out = []
        p_inters_out = [ [] for _ in range(len(persons))]
        for p1_idx, p1_joints in enumerate(persons):
            for offset, p2_joints in enumerate(persons[p1_idx+1:]):
                p2_idx = p1_idx + offset + 1
                rels_name = rel_type+"_p{:0>2}-p{:0>2}".format(p1_idx,p2_idx)
                inter_avg = make_connections('p1_p2_all', g_theta_model, 
                               p1_joints, p2_joints, name_suffix=rels_name)
                p_inters_out[p1_idx].append(inter_avg)
                p_inters_out[p2_idx].append(inter_avg)
            rels_name = rel_type+"_p{:0>2}".format(p1_idx)
            p_inters_avg = pool_relations(p_inters_out[p1_idx], 'avg', 
                                          name_suffix=rels_name)
            inters_out.append(p_inters_avg)
        rel_out = inters_out
    # elif rel_type == 'inter-graph':
    elif rel_type.startswith('inter-graph') and not rel_type.endswith('att'):
        #### [inter-graph]
        ## inter directional with graph
        # Get Adjacency Matrix
        graph_idx = int(rel_type[-1])
        adj_mat = G[graph_idx]
        
        inters_out = []
        for p1_idx, p1_connections in enumerate(adj_mat):
            p1_joints = persons[p1_idx]
            p1_neighbours = [i for i,v in enumerate(p1_connections) if v==1]
            p_inters_out = []
            # for p2_idx, p2_joints in enumerate(persons):
            for p2_idx in p1_neighbours:
                p2_joints = persons[p2_idx]
                if p1_idx != p2_idx:
                    rels_name = rel_type+"_p{:0>2}-p{:0>2}".format(p1_idx,p2_idx)
                    inter_avg = make_connections('p1_p2_all', g_theta_model, 
                                   p1_joints, p2_joints, name_suffix=rels_name,
                                   skip_pool=True)
                    # p_inters_out.append(inter_avg)
                    p_inters_out += inter_avg
            rels_name = rel_type+"_p{:0>2}".format(p1_idx)
            p_inters_avg = pool_relations(p_inters_out, 'avg', 
                                          name_suffix=rels_name)
            inters_out.append(p_inters_avg)
        rel_out = inters_out
    elif rel_type.startswith('inter-graph') and rel_type.endswith('att'):
        #### [inter-graph-x-att]
        ## inter directional with graph
        # Get Adjacency Matrix
        graph_idx = int(rel_type.split('-')[-2])
        adj_mat = G[graph_idx]
        
        inters_out = []
        for p1_idx, p1_connections in enumerate(adj_mat):
            p1_joints = persons[p1_idx]
            p1_neighbours = [i for i,v in enumerate(p1_connections) if v==1]
            p_inters_out = []
            # for p2_idx, p2_joints in enumerate(persons):
            for p2_idx in p1_neighbours:
                p2_joints = persons[p2_idx]
                if p1_idx != p2_idx:
                    rels_name = rel_type+"_p{:0>2}-p{:0>2}".format(p1_idx,p2_idx)
                    inter_avg = make_connections('p1_p2_all', g_theta_model, 
                                   p1_joints, p2_joints, name_suffix=rels_name,
                                   skip_pool=False)
                                   # skip_pool=True)
                    p_inters_out.append(inter_avg)
                    # p_inters_out += inter_avg
            # rels_name = rel_type+"_p{:0>2}".format(p1_idx)
            # p_inters_avg = pool_relations(p_inters_out, 'avg', 
            #                               name_suffix=rels_name)
            # inters_out.append(p_inters_avg)
            inters_out.append(p_inters_out)
        inters_att_out = pool_relations(inters_out, 'att-luong_inter', 
                                      name_suffix=rel_type)
        rel_out = inters_att_out
    elif rel_type == 'intra' or rel_type == 'indivs':
        ### intra/indivs
        if persons == []:
            indiv1_avg = make_connections('p1_p1_all', g_theta_model, 
                p1_joints, [])
            
            if p2_joints != []:
                indiv2_avg = make_connections('p1_p1_all', g_theta_model, 
                                              p2_joints, [])
                rel_out = Concatenate()([indiv1_avg, indiv2_avg])
            else:
                rel_out = indiv1_avg
        else:
            indivs_out = []
            for p_idx, p_joints in enumerate(persons):
                rels_name = rel_type+"_p{:0>2}".format(p_idx)
                indiv_avg = make_connections('p1_p1_all', g_theta_model, 
                     p_joints, [], name_suffix=rels_name)
                indivs_out.append(indiv_avg)
            rel_out = indivs_out
            # if len(indivs_out) == 1:
            #     rel_out = indivs_out[0]
            # else:
            #     rel_out = Average()(indivs_out)
    elif rel_type == 'inter-objs':
        ### inter-objs
        inters_out = []
        for p_idx, p_joints in enumerate(persons):
            if len(extra_objs) == 1:
                o_coords = extra_objs[0]
                rels_name = rel_type+"_p{:0>2}".format(p_idx)
                inter_avg = make_connections('p1_p2_all', g_theta_model, 
                                     p_joints, o_coords, name_suffix=rels_name)
                inters_out.append(inter_avg)
            else:
                p_inters_out = []
                for o_idx, o_coords in enumerate(extra_objs):
                    rels_name = rel_type+"_p{:0>2}-o{:0>2}".format(p_idx, o_idx)
                    inter_avg = make_connections('p1_p2_all', g_theta_model, 
                                         p_joints, o_coords, name_suffix=rels_name)
                    p_inters_out.append(inter_avg)
                rels_name = rel_type+"_p{:0>2}".format(p_idx)
                p_inters_avg = pool_relations(p_inters_out, 'avg', 
                                              name_suffix=rels_name)
                inters_out.append(inter_avg)
        rel_out = inters_out
    elif rel_type == 'objs':
        ### objs
        inters_out = []
        for idx, o1_coords in enumerate(extra_objs[:-1]):
            for o2_coords in extra_objs[idx+1:]:
                inter_avg = make_connections('inter', g_theta_model, 
                                    o1_coords, o2_coords)
                inters_out.append(inter_avg)
        rel_out = inters_out
        # if len(inters_out) == 1:
        #     rel_out = inters_out[0]
        # else:
        #     rel_out = Average()(inters_out)
    elif rel_type == 'inter_and_indivs':
        # All joints from person1 connected to all joints of person2, and back
        for object_i in p1_joints:
            for object_j in p2_joints:
                g_theta_outs.append(g_theta_model([object_i, object_j]))
        for object_i in p2_joints:
            for object_j in p1_joints:
                g_theta_outs.append(g_theta_model([object_i, object_j]))
        
        # All joints from person1 connected to all other joints of itself
        for idx, object_i in enumerate(p1_joints):
            for object_j in p1_joints[idx+1:]:
            # for object_j in p1_joints[idx:]:
                g_theta_outs.append(g_theta_model([object_i, object_j]))
                
        # All joints from person2 connected to all other joints of itself
        for idx, object_i in enumerate(p2_joints):
            for object_j in p2_joints[idx+1:]:
            # for object_j in p2_joints[idx:]:
                g_theta_outs.append(g_theta_model([object_i, object_j]))
        
        # rel_out = Average()(g_theta_outs)
        rel_out = pool_relations(g_theta_outs, 'avg', name_suffix=name_suffix)
    elif rel_type == 'p1_p2_all':
        # All joints from person1 connected to all joints of person2
        for object_i in p1_joints:
            for object_j in p2_joints:
                g_theta_outs.append(g_theta_model([object_i, object_j]))
        # rel_out = Average()(g_theta_outs)
        if not skip_pool:
            rel_out = pool_relations(g_theta_outs, 'avg', 
                                     name_suffix=name_suffix)
        else:
            rel_out = g_theta_outs
    elif rel_type == 'p1_p1_all':
        # All joints from person1 connected to all other joints of itself
        for idx, object_i in enumerate(p1_joints):
            for object_j in p1_joints[idx+1:]:
            # for object_j in p1_joints[idx:]:
                g_theta_outs.append(g_theta_model([object_i, object_j]))
        # rel_out = Average()(g_theta_outs)
        rel_out = pool_relations(g_theta_outs, 'avg', name_suffix=name_suffix)
    elif rel_type == 'p1_p1_all_bidirectional':
        # All joints from person1 connected to all other joints of itself, and back
        rel_out = make_connections(
            'p1_p2_all_bidirectional', g_theta_model, p1_joints, p1_joints)
    elif rel_type == 'p2_p2_all_bidirectional':
        # All joints from person2 connected to all other joints of itself, and back
        rel_out = make_connections(
            'p1_p2_all_bidirectional', g_theta_model, p2_joints, p2_joints)
    elif rel_type == 'p1_p1_all-p2_p2_all':
        # All joints from person1 connected to all other joints of itself
        for idx, object_i in enumerate(p1_joints):
            for object_j in p1_joints[idx+1:]:
            # for object_j in p1_joints[idx:]:
                g_theta_outs.append(g_theta_model([object_i, object_j]))
        for idx, object_i in enumerate(p2_joints):
            for object_j in p2_joints[idx+1:]:
            # for object_j in p1_joints[idx:]:
                g_theta_outs.append(g_theta_model([object_i, object_j]))
        # rel_out = Average()(g_theta_outs)
        rel_out = pool_relations(g_theta_outs, 'avg', name_suffix=name_suffix)
    else:
        raise ValueError("Invalid rel_type:", rel_type)
    
    return rel_out

def expand_and_concat(tensors_list):
    # expanded = [ K.expand_dims(tensor, 1) for tensor in tensors_list ]
    # return K.concatenate(expanded, 1)
    return K.stack(tensors_list, axis=1)

def pool_relations(relations, mode='avg', name_suffix='', query_rels=None, 
                   att_dr=0):
    ## Using att_dr argument for drop_rate arg in other pools also
    
    # if name_suffix is not None and not name_suffix.startswith('_'):
    if name_suffix != '' and not name_suffix.startswith('_'):
        name_suffix = '_' + name_suffix
    
    # if not isinstance(relations, list):
    #     # No need to pool
    #     return relations
        
    if len(relations) == 1:
        # No need to pool
        return relations[0]
    
    if mode == 'avg':
        ### Dropout rels before avg
        if att_dr > 0:
            # print("Warning: Using NEW indiv_rels_dr.")            
            indiv_dr_layer = Dropout(att_dr, noise_shape=(1,1), 
                                     name='full_rels_dr'+name_suffix)
            # print("Warning: Activating dropout for prediction also.")
            # relations = [ indiv_dr_layer(indiv_rels, training=True) 
            relations = [ indiv_dr_layer(indiv_rels) 
                         for indiv_rels in relations]
            
        pooled_rels = Average(name=mode+name_suffix)(relations)
    elif mode == 'sum':
        pooled_rels = Add(name=mode+name_suffix)(relations)
    elif mode == 'max':
        pooled_rels = Maximum(name=mode+name_suffix)(relations)
    elif mode == 'conc':
        pooled_rels = Concatenate(name=mode+name_suffix)(relations)
    elif mode == 'att-q_avg': # Attention
        value = Lambda(expand_and_concat, 
                           name='value'+name_suffix)(relations)
        query = Lambda(lambda x: K.mean(x, axis=1, keepdims=True), 
                            name='query'+name_suffix)(value)
        key = value
        
        att = AdditiveAttention(use_scale=True, causal=False, 
                        name='attention'+name_suffix, dropout=att_dr)
        att_out = att([query, value]) 
        avg_att = GlobalAveragePooling1D()(att_out)
        pooled_rels = avg_att
    elif mode == 'att-q_ones': # Attention
        value = Lambda(expand_and_concat, 
                           name='value'+name_suffix)(relations)
        query = K.expand_dims(tf.ones_like(relations[0]), axis=1)
        # print("WARNING: Setting Query Attention as zeros!")
        # query = K.expand_dims(tf.zeros_like(relations[0]), axis=1)
        key = value
        
        att = AdditiveAttention(use_scale=True, causal=False, 
                        name='attention'+name_suffix, dropout=att_dr)
        
        att_out = att([query, value]) 
        avg_att = GlobalAveragePooling1D()(att_out)
        pooled_rels = avg_att
    elif mode == 'att-q_dense': # Attention
        ## Attention Bahdanau with Dense
        value = Lambda(expand_and_concat, 
                           name='value'+name_suffix)(relations)
        
        avg = Lambda(lambda x: K.mean(x, axis=1, keepdims=True), 
                            name='avg'+name_suffix)(value)
        
        if isinstance(att_dr, list):
            query_dr, scores_dr = att_dr
        else:
            query_dr = 0
            scores_dr = att_dr
        
        if query_dr > 0:
            avg = Dropout(query_dr)(avg)
        
        kernel_init = 'ones' # 'ones' 'zeros' 'glorot_uniform'
        # kernel_init = get_kernel_init('TruncatedNormal', param=0.045) # default IRN
        query = Dense(relations[0].shape[-1], name='query'+name_suffix, 
                      kernel_initializer=kernel_init)(avg)
        
        # print(query, value, key)
        att = AdditiveAttention(use_scale=True, causal=False, 
                        name='attention'+name_suffix, dropout=scores_dr)
        att_out = att([query, value]) 
        avg_att = GlobalAveragePooling1D()(att_out)
        pooled_rels = avg_att
    elif mode == 'att-q_dense2': # Attention
        ## Attention Bahdanau with Tanh Dense 
        value = Lambda(expand_and_concat, 
                           name='value'+name_suffix)(relations)
        
        avg = Lambda(lambda x: K.mean(x, axis=1, keepdims=True), 
                            name='avg'+name_suffix)(value)
        
        if isinstance(att_dr, list):
            query_dr, scores_dr = att_dr
        else:
            query_dr = 0
            scores_dr = att_dr
        
        if query_dr > 0:
            avg = Dropout(query_dr)(avg)
        
        # kernel_init = 'ones'
        # kernel_init = 'zeros'
        # kernel_init = 'glorot_uniform'
        # kernel_init = get_kernel_init('TruncatedNormal', param=0.045) # default IRN
        kernel_init = get_kernel_init('TruncatedNormal', param=0.01)
        # kernel_init = initializers.TruncatedNormal(mean=1., stddev=1.)
        print("kernel_init:", kernel_init)
        # activation = None
        activation = 'tanh'
        print("activation:", activation)
        query = Dense(relations[0].shape[-1], name='query'+name_suffix, 
                      activation=activation,
                      kernel_initializer=kernel_init)(avg)
        # key = value
        key = tf.keras.layers.Activation('tanh')(value)
        print("Using key:", key)
        
        # print(query, value, key)
        att = AdditiveAttention(use_scale=True, causal=False, 
                        name='attention'+name_suffix, dropout=scores_dr)
        att_out = att([query, value, key]) 
        avg_att = GlobalAveragePooling1D()(att_out)
        pooled_rels = avg_att
    elif mode == 'att-luong': # Luong-style
        #### [att-luong]
        ## Attention Luong with Dense
        if isinstance(att_dr, list):
            if len(att_dr) == 2:
                key_dr, scores_dr = att_dr
                query_dr = 0
            else:
                query_dr, key_dr, scores_dr = att_dr
        else:
            query_dr = 0
            key_dr = 0
            scores_dr = att_dr
        
        ## Remove if to go back to previous scores dr inside att
        if scores_dr > 0:
            # print("Warning: Using NEW indiv_rels_dr.")            
            indiv_dr_layer = Dropout(scores_dr, noise_shape=(1,1), 
                                     name='full_rels_dr'+name_suffix)
            # print("Warning: Activating dropout for prediction also.")
            # relations = [ indiv_dr_layer(indiv_rels, training=True) 
            relations = [ indiv_dr_layer(indiv_rels) 
                         for indiv_rels in relations]
            scores_dr = 0 # Setting to 0 so it will not be used inside att
            
        value = Lambda(expand_and_concat, 
                           name='value'+name_suffix)(relations)
        
        ## Query as Dense over avg 
        # avg = Lambda(lambda x: K.mean(x, axis=1, keepdims=True), 
        #                     name='avg'+name_suffix)(value)
        # # kernel_init = 'ones' # 'ones' 'zeros' 'glorot_uniform'
        # # kernel_init = get_kernel_init('TruncatedNormal',param=0.01)
        # print("kernel_init:", kernel_init)
        # activation = 'tanh' # [None] 'tanh'
        # print("activation:", activation)
        # query = Dense(relations[0].shape[-1], name='query'+name_suffix, 
        #               activation=activation,
        #               kernel_initializer=kernel_init)(avg)
        # key = value
        
        ## Query as Ones
        # query = K.expand_dims(tf.ones_like(relations[0]), axis=1)
        
        ## Query as contant to apply avg
        # # create constant with 1/500 (so it would be simply the average)
        # avg_factor = 1/relations[0].shape[-1]
        # ones = tf.ones_like(relations[0])
        # query = K.expand_dims(tf.scalar_mul(avg_factor, ones), axis=1)
        
        ## Query as learnable parameter
        # init_val = [[avg_factor] * relations[0].shape[-1]]
        # query_var = tf.Variable(initial_value=init_val, trainable=True)
        class Query(tf.keras.layers.Layer):
            def __init__(self, **kwargs):
                super(Query, self).__init__(**kwargs)
            
            def build(self, input_shape):
                init_vals = [[1/input_shape[-1]] * input_shape[-1]]
                self.w = tf.Variable(initial_value=init_vals, trainable=True)
            
            def call(self, inputs):
                # return self.w
                # Workaround to add None to shape (batch size)
                exp_inputs = tf.expand_dims(inputs, axis=1)
                # return exp_inputs - exp_inputs + self.w
                return tf.zeros_like(exp_inputs) + self.w
        
        query = Query(name='query'+name_suffix)(relations[0])
        if query_dr > 0:
            query = Dropout(query_dr, name='dropout_query')(query)
        
        # key = tf.keras.layers.Activation('tanh')(value)
        # kernel_init = 'ones' # 'ones' 'zeros' 'glorot_uniform'
        kernel_init = get_kernel_init('TruncatedNormal',param=0.045)
        # kernel_init = get_kernel_init('TruncatedNormal',param=0.01)
        activation = 'tanh' # [None] 'tanh' 'sigmoid'
        # print("kernel_init:", kernel_init)
        # print("activation:", activation)
        key_dense = Dense(relations[0].shape[-1], name='key_transform'+name_suffix, 
                         activation=activation, kernel_initializer=kernel_init)
        
        ## Key Dropout Before Dense
        if key_dr == 0:
            key_list = [ key_dense(ind_rels) for ind_rels in relations ]
        else:
            # print("Previous key dropout order: dr BEFORE key_dense")
            key_dr_layer = Dropout(key_dr)
            key_list = [ key_dense(key_dr_layer(ind_rels)) 
                        for ind_rels in relations ]
        key = Lambda(expand_and_concat, 
                      name='key'+name_suffix)(key_list)
        
        ## Key Dropout after Dense
        # key_list = [ key_dense(ind_rels) for ind_rels in relations ]
        # key = Lambda(expand_and_concat, 
        #               name='key'+name_suffix)(key_list)
        # if key_dr > 0:
        #     print("Inverted key dropout order: dr after key_dense")
        #     key = Dropout(key_dr, name='dropout_key')(key)
        
        # print("Applying math.abs to key output")
        # key = Lambda(lambda x: tf.math.abs(K.stack(x, axis=1)), 
        #              name='key'+name_suffix)(key_list)
        
        # print(query, value, key)
        # print("Warning: Setting use_scale as True!")
        att = Attention(use_scale=False, causal=False, 
                        name='attention'+name_suffix, dropout=scores_dr)
        att_out = att([query, value, key])
        # print("Warning: Activating dropout for prediction also.")
        # att_out = att([query, value, key], training=True)
        avg_att = GlobalAveragePooling1D()(att_out)
        pooled_rels = avg_att
    elif mode == 'att-luong_inter': # Luong-style
        #### [att-luong_inter]
        ## Attention Luong of person interactions with Dense
        ## Only to be used by rel_type = 'inter*-att'
        if isinstance(att_dr, list):
            if len(att_dr) == 2:
                key_dr, scores_dr = att_dr
                query_dr = 0
            else:
                query_dr, key_dr, scores_dr = att_dr
        else:
            query_dr = 0
            key_dr = 0
            scores_dr = att_dr
        
        #### WIP Hard-coded values
        print("Warning: Hard-coded values")
        # query_dr, key_dr, scores_dr = 0.1, 0.1, 0.1
        query_dr, key_dr, scores_dr = 0.25, 0.25, 0.25
        print("inter_att_dr:", query_dr, key_dr, scores_dr)
        
        class Query(tf.keras.layers.Layer):
            def __init__(self, **kwargs):
                super(Query, self).__init__(**kwargs)
            
            def build(self, input_shape):
                init_vals = [[1/input_shape[-1]] * input_shape[-1]]
                self.w = tf.Variable(initial_value=init_vals, trainable=True)
            
            def call(self, inputs):
                # return self.w
                # Workaround to add None to shape (batch size)
                exp_inputs = tf.expand_dims(inputs, axis=1)
                # return exp_inputs - exp_inputs + self.w
                return tf.zeros_like(exp_inputs) + self.w
        
        query = Query(name='query_inters')(relations[0][0])
        if query_dr > 0:
            query = Dropout(query_dr, name='dropout_query_inters')(query)
        
        # value = Lambda(expand_and_concat, 
        #                    name='value'+name_suffix)(relations)
        
        kernel_init = get_kernel_init('TruncatedNormal',param=0.045)
        activation = 'tanh' # [None] 'tanh' 'sigmoid'
        key_dense = Dense(relations[0][0].shape[-1], 
                         # name='key_inters_transform'+name_suffix, 
                         name='key_inters_transform', 
                         activation=activation, kernel_initializer=kernel_init)
        
        ## Key Dropout Before Dense
        if key_dr == 0:
            # key_list = [ key_dense(ind_rels) for ind_rels in relations ]
            key_list = [ [key_dense(p_inter) for p_inter in p_inters]
                          for p_inters in relations ]
        else:
            # print("Previous key dropout order: dr BEFORE key_dense")
            key_dr_layer = Dropout(key_dr)
            # key_list = [ key_dense(key_dr_layer(ind_rels)) 
            #             for ind_rels in relations ]
            key_list = [ [key_dense(key_dr_layer(p_inter)) 
                              for p_inter in p_inters ]
                          for p_inters in relations ]
        
        ## Remove if to go back to previous scores dr inside att
        if scores_dr > 0:
            # print("Warning: Using NEW inters_dr_layer.")            
            inters_dr_layer = Dropout(scores_dr, noise_shape=(1,1), 
                                     name='full_inters_dr')
            # print("Warning: Activating dropout for prediction also.")
            dr_relations = [ [inters_dr_layer(p_inter) for p_inter in p_inters]
                          for p_inters in relations ]
            relations = dr_relations
            scores_dr = 0 # Setting to 0 so it will not be used inside att
        
        att = Attention(use_scale=False, causal=False, 
                        # name='inters_attention'+name_suffix, dropout=scores_dr)
                        name='inters_attention', dropout=scores_dr)
        
        pooled_rels = []
        # for i, indiv_query in enumerate(query_list):
        for i in range(len(relations)):
            p_inters = relations[i]
            p_inters_key = key_list[i]
            
            value = Lambda(expand_and_concat, 
                    # name='value_p{:0>2}'.format(i+1) + name_suffix)(p_inters)
                    name='value_inters_p{:0>2}'.format(i+1))(p_inters)
            
            key = Lambda(expand_and_concat, 
                    # name='key_p{:0>2}'.format(i+1) + name_suffix)(p_inters_key)
                    name='key_inters_p{:0>2}'.format(i+1))(p_inters_key)
            
            # query = Lambda(lambda x: K.expand_dims(x, 1), 
            #         name='query_p{:0>2}'.format(i) + name_suffix)(indiv_query)
            
            att_out = att([query, value, key]) 
            # pooled_rels.append(tf.squeeze(att_out, axis=1))
            #### TODO Need testing to see if weights load, using lambda to rename
            pooled_rels.append(Lambda(lambda x: tf.squeeze(x, axis=1),
                  name='avg'+name_suffix+'_p{:0>2}'.format(i))(att_out) )
    elif mode == 'indiv_att-luong': # Luong-style
        #### [indiv_att-luong]
        ## Indiv Attention Luong with Dense, scores_dr before att (indiv_full_rels_dr)
        if isinstance(att_dr, list):
            if len(att_dr) == 2:
                key_dr, scores_dr = att_dr
                query_dr = 0
            else:
                query_dr, key_dr, scores_dr = att_dr
        else:
            query_dr = 0
            key_dr = 0
            scores_dr = att_dr
        
        # kernel_init = 'ones' # 'ones' 'zeros' 'glorot_uniform'
        kernel_init = get_kernel_init('TruncatedNormal',param=0.045)
        activation = 'tanh' # [None] 'tanh' 'sigmoid'
        key_dense = Dense(relations[0].shape[-1], name='indiv_key_transform'+name_suffix, 
                         activation=activation, kernel_initializer=kernel_init)
        
        ## Key Dropout Before Dense
        if key_dr == 0:
            key_list = [ key_dense(ind_rels) for ind_rels in relations ]
            # key_list = [ key_dense(ind_rels) for ind_rels in dr_relations ]
        else:
            # print("Previous key dropout order: dr BEFORE key_dense")
            key_dr_layer = Dropout(key_dr)
            key_list = [ key_dense(key_dr_layer(ind_rels)) 
                        for ind_rels in relations ]
                        # for ind_rels in dr_relations ]
        
        units = relations[0].shape[-1]
        query_layer = Dense(units, name='indiv_query_transform'+name_suffix,
                      kernel_initializer='zeros', 
                      bias_initializer=tf.keras.initializers.constant(1/units))
        # print("Warning: indiv query from rels directly.") # 1) w/ relations
        # print("Warning: indiv query from key transform.") # 2) w/ key_list
        if query_dr == 0:
            # query_list = [ query_layer(ind_rels) for ind_rels in relations ]
            query_list = [ query_layer(ind_rels) for ind_rels in key_list ]
        else:
            query_dr_layer = Dropout(query_dr)
            query_list = [ query_layer(query_dr_layer(ind_rels)) 
                        # for ind_rels in relations ]
                        for ind_rels in key_list ]
        
        ## Remove if to go back to previous scores dr inside att
        if scores_dr > 0:
            # print("Warning: Using NEW indiv_full_rels_dr.") # indiv_rels_dr
            indiv_dr_layer = Dropout(scores_dr, noise_shape=(1,1,1), 
                                      name='indiv_full_rels_dr'+name_suffix)
            
            stack = Lambda(lambda x: tf.stack(x, axis=1), name='stack')
            unstack = Lambda(lambda x: tf.unstack(x, axis=1), name='unstack')
            
            # print("Warning: Activating dropout for prediction also.")
            dr_rels_and_keys = [
                # tf.unstack(indiv_dr_layer(tf.stack([indiv_rels, indiv_key], 
                #                                    axis=1)) , axis=1)
                unstack(indiv_dr_layer(stack([indiv_rels, indiv_key])))
                # unstack(indiv_dr_layer(stack([indiv_rels, indiv_key]), training=True))
                for indiv_rels, indiv_key in zip(relations, key_list) ]
            
            dr_relations = [ indiv_dr[0] for indiv_dr in dr_rels_and_keys]
            dr_key_list = [ indiv_dr[1] for indiv_dr in dr_rels_and_keys]
            
            scores_dr = 0 # Setting to 0 so it will not be used inside att
        else:
            dr_relations = relations
            dr_key_list = key_list
        
        # print(query, value, key)
        # print("Warning: Setting indiv use_scale as True!")
        att = Attention(use_scale=False, causal=False, 
                        name='indiv_attention'+name_suffix, dropout=scores_dr)
        
        # print("Warning: ALL rels att pool.")
        # query = Lambda(expand_and_concat, # a.k.a. stack
        #               name='indiv_query'+name_suffix)(query_list)
        # value = Lambda(expand_and_concat, 
        #                     name='indiv_value'+name_suffix)(relations)
        # key = Lambda(expand_and_concat, 
        #               name='indiv_key'+name_suffix)(key_list)
        # att_out = att([query, value, key]) 
        # pooled_rels = tf.unstack(att_out, axis=1)
                
        # print("Warning: others rels att pool.")
        pooled_rels = []
        for i, indiv_query in enumerate(query_list):
            # others_rels = relations.copy()
            others_rels = dr_relations.copy()
            others_rels.pop(i)
            value = Lambda(expand_and_concat, 
                    name='value_p{:0>2}'.format(i) + name_suffix)(others_rels)
            
            # others_keys = key_list.copy()
            others_keys = dr_key_list.copy()
            others_keys.pop(i)
            key = Lambda(expand_and_concat, 
                    name='key_p{:0>2}'.format(i) + name_suffix)(others_keys)
            
            query = Lambda(lambda x: K.expand_dims(x, 1), 
                    name='query_p{:0>2}'.format(i) + name_suffix)(indiv_query)
            
            att_out = att([query, value, key]) 
            pooled_rels.append(tf.squeeze(att_out, axis=1))
    elif mode == 'indiv_att-luong3': # Luong-style
        #### [indiv_att-luong3]    
        ## Indiv Attention Luong with Dense, and scr_dr after att
        if isinstance(att_dr, list):
            if len(att_dr) == 2:
                key_dr, scores_dr = att_dr
                query_dr = 0
            else:
                query_dr, key_dr, scores_dr = att_dr
        else:
            query_dr = 0
            key_dr = 0
            scores_dr = att_dr
        
        # kernel_init = 'ones' # 'ones' 'zeros' 'glorot_uniform'
        kernel_init = get_kernel_init('TruncatedNormal',param=0.045)
        activation = 'tanh' # [None] 'tanh' 'sigmoid'
        key_dense = Dense(relations[0].shape[-1], name='indiv_key_transform'+name_suffix, 
                         activation=activation, kernel_initializer=kernel_init)
        
        ## Key Dropout Before Dense
        if key_dr == 0:
            key_list = [ key_dense(ind_rels) for ind_rels in relations ]
        else:
            # print("Previous key dropout order: dr BEFORE key_dense")
            key_dr_layer = Dropout(key_dr)
            key_list = [ key_dense(key_dr_layer(ind_rels)) 
                        for ind_rels in relations ]
        
        units = relations[0].shape[-1]
        query_layer = Dense(units, name='indiv_query_transform'+name_suffix,
                      kernel_initializer='zeros', 
                      bias_initializer=tf.keras.initializers.constant(1/units))
        # print("Warning: indiv query from rels directly.") # 1) w/ relations
        # print("Warning: indiv query from key transform.") # 2) w/ key_list
        if query_dr == 0:
            # query_list = [ query_layer(ind_rels) for ind_rels in relations ]
            query_list = [ query_layer(ind_rels) for ind_rels in key_list ]
        else:
            query_dr_layer = Dropout(query_dr)
            query_list = [ query_layer(query_dr_layer(ind_rels)) 
                        # for ind_rels in relations ]
                        for ind_rels in key_list ]
        
        # print(query, value, key)
        att = Attention(use_scale=False, causal=False, 
                        name='indiv_attention'+name_suffix, dropout=scores_dr)
        
        # print("Warning: ALL rels att pool.")
        # query = Lambda(expand_and_concat, # a.k.a. stack
        #               name='indiv_query'+name_suffix)(query_list)
        # value = Lambda(expand_and_concat, 
        #                     name='indiv_value'+name_suffix)(relations)
        # key = Lambda(expand_and_concat, 
        #               name='indiv_key'+name_suffix)(key_list)
        # att_out = att([query, value, key]) 
        # pooled_rels = tf.unstack(att_out, axis=1)
                
        # print("Warning: others rels att pool.")
        pooled_rels = []
        for i, indiv_query in enumerate(query_list):
            others_rels = relations.copy()
            others_rels.pop(i)
            value = Lambda(expand_and_concat, 
                    name='value_p{:0>2}'.format(i) + name_suffix)(others_rels)
            
            others_keys = key_list.copy()
            others_keys.pop(i)
            key = Lambda(expand_and_concat, 
                    name='key_p{:0>2}'.format(i) + name_suffix)(others_keys)
            
            query = Lambda(lambda x: K.expand_dims(x, 1), 
                    name='query_p{:0>2}'.format(i) + name_suffix)(indiv_query)
            
            att_out = att([query, value, key]) 
            pooled_rels.append(tf.squeeze(att_out, axis=1))
    elif mode == 'att': # Attention with wrong query
        conc_rels = Lambda(expand_and_concat, 
                           name='conc_rels'+name_suffix)(relations)
        
        # if att_dr > 0:
        #     conc_rels = Dropout(att_dr)(conc_rels)
        #     # conc_rels = SpatialDropout1D(att_dr)(conc_rels)
        
        value = conc_rels
        
        if query_rels is None:
            query = conc_rels
            key = conc_rels
        else:
            ## Not working because of different dims between value and query
            conc_rels_query = Lambda(expand_and_concat, 
                               name='conc_rels_query'+name_suffix)(query_rels)
            query = conc_rels_query
            key = conc_rels_query
        # print(query)
        # print(value)
        # print(key)
        
        # Attention (Luong-style) | AdditiveAttention (Bahdanau-style)
        att = AdditiveAttention(use_scale=True, causal=False, 
                        name='attention'+name_suffix, dropout=att_dr)
        att_out = att([query, value, key]) 
        avg_att = GlobalAveragePooling1D()(att_out)
        pooled_rels = avg_att
    else:
        raise ValueError("Invalid pooling mode:", mode)
    
    return pooled_rels

def create_relationships(rel_type, g_theta_model, p1_joints, p2_joints, 
             persons=[], extra_objs=[], pool_mode='avg', rels_dr=0, att_dr=0,
             out_rels=False):
    #### TODO Insert inter_att_dr here?
    g_theta_outs = make_connections(rel_type, g_theta_model, p1_joints, 
                            p2_joints, persons=persons, extra_objs=extra_objs)
    
    if rels_dr > 0:
        g_theta_outs = [ Dropout(rels_dr)(rels_out) 
                        for rels_out in g_theta_outs ]
    
    if pool_mode is not None:
        rel_out = pool_relations(g_theta_outs, mode=pool_mode, 
                                 name_suffix=rel_type, att_dr=att_dr)
    else:
        rel_out = g_theta_outs
    
    if out_rels:
        rel_out = [rel_out, g_theta_outs]
    
    return rel_out

def multiple_rels(rels_type, person1_joints, persons, extra_objs, pool_mode, 
                  query_rels_type=None, att_dr=0, **g_theta_kwargs):
    
    rels_type_list = rels_type.split('+')
    mult_rels_out = []
    for rel_type in rels_type_list:
        g_theta_model = g_theta(model_name="g_theta_"+rel_type, 
                                **g_theta_kwargs)
        rels_out  = create_relationships(rel_type, g_theta_model, 
            person1_joints, [], persons, extra_objs, pool_mode=None)
        mult_rels_out.append(rels_out)
    
    # Merged rels per person
    merged_rels = []
    for person_idx in range(len(persons)):
        person_rels = [ rels_out[person_idx] for rels_out in mult_rels_out ]
        merged_rels.append(Concatenate()(person_rels))
    
    if query_rels_type is None:
        query_rels = None
    else:
        query_rels = mult_rels_out[rels_type_list.index(query_rels_type)]
    rel_out = pool_relations(merged_rels, mode=pool_mode, 
                             query_rels=query_rels, att_dr=att_dr)
    
    # rel_out = pool_relations(mult_rels_out[0])
    
    return rel_out

def create_top_indiv(pooled_rels, rels_outs, rel_type, kernel_init, drop_rate=0, 
                 fc_units=[500,100,100], fc_drop=False, indiv_conc_pool=False,
                 indiv_pool_mode='same', att_dr=0):
    
    if indiv_conc_pool:
        tpl = 'conc_'+rel_type+'_p{:0>2}'
        
        ## Append same pooled rels used for grp classification (avg, att...)
        if indiv_pool_mode == 'same':
            conc_rels_outs = [ 
                Concatenate(name=tpl.format(i))([pooled_rels, indiv_rels]) 
                for i, indiv_rels in enumerate(rels_outs) ]
        elif indiv_pool_mode == 'others_avg':
            ## Averaging all rels, regardless of the pool for grp class.
            # pooled_rels = pool_relations(rels_outs, 'avg', 'rels')
            # conc_rels_outs = [ 
            #     Concatenate(name=tpl.format(i))([pooled_rels, indiv_rels]) 
            #     for i, indiv_rels in enumerate(rels_outs) ]
        
            ## Averaging all rels from the other players
            conc_rels_outs = []
            for i, indiv_rels in enumerate(rels_outs):
                other_rels_outs = rels_outs.copy()
                other_rels_outs.pop(i)
                pooled_rels = pool_relations(other_rels_outs, 'avg', 
                                              'other_rels_p{:0>2}'.format(i))
                conc_rels_out = Concatenate(name=tpl.format(i))([pooled_rels, 
                                                                  indiv_rels]) 
                conc_rels_outs.append(conc_rels_out)
        else: # TODO Only for startswith('att')?
            indiv_pool = pool_relations(rels_outs, mode=indiv_pool_mode, 
                                  name_suffix=rel_type, att_dr=att_dr)
            conc_rels_outs = [ 
                Concatenate(name=tpl.format(i))([indiv_pool[i], indiv_rels]) 
                for i, indiv_rels in enumerate(rels_outs) ]
        
        rels_outs = conc_rels_outs
    
    input_shape = rels_outs[0].shape[-1:]
    input_top = Input(shape=input_shape)
    out_top = create_top(input_top, kernel_init, fc_units=fc_units, 
                           drop_rate=drop_rate, fc_drop=fc_drop)
    top_indiv = Model(inputs=input_top, outputs=out_top, name="f_phi_indivs")
    
    top_indivs_out = []
    for indiv_rels in rels_outs:
        top_indivs_out.append(top_indiv(indiv_rels))
    
    return top_indivs_out

def create_top(input_top, kernel_init, drop_rate=0, fc_units=[500,100,100], 
        fc_drop=False):
    x = Dropout(drop_rate)(input_top)
    
    x = Dense(fc_units[0], activation='relu', kernel_initializer=kernel_init, 
        name="f_phi_fc1")(x)
    if fc_drop: x = Dropout(drop_rate)(x)
    x = Dense(fc_units[1], activation='relu', kernel_initializer=kernel_init, 
        name="f_phi_fc2")(x)
    if fc_drop: x = Dropout(drop_rate)(x)
    x = Dense(fc_units[2], activation='relu', kernel_initializer=kernel_init, 
        name="f_phi_fc3")(x)
    
    return x

def create_input(num_indivs, num_joints, object_shape, num_extra_objs=0,
                 num_coords_obj=3):
    persons = []
    for ind_idx in range(num_indivs):
        person_joints = []
        for obj_idx in range(num_joints):
            obj = Input(shape=object_shape,
                        name="p{:0>2}_obj{:0>2}".format(ind_idx+1, obj_idx+1))
            person_joints.append(obj)
        persons.append(person_joints)
        
    extra_objs = []
    for x_obj_idx in range(num_extra_objs):
        obj_coords = []
        for obj_idx in range(num_coords_obj):
            obj = Input(shape=object_shape,
                        name="o{}_obj{}".format(x_obj_idx+1, obj_idx+1))
            obj_coords.append(obj)
        extra_objs.append(obj_coords)
    
    return persons, extra_objs

def f_phi(num_objs, object_shape, rel_type, kernel_init,fc_units=[500,100,100],
        drop_rate=0, fuse_type=None, fc_drop=False, individual=False,
        num_indivs=2, num_extra_objs=0, num_coords_obj=3, pool_mode='avg', 
        att_dr=0, rels_dr=0, indiv_out=False, indiv_conc_pool=False, 
        indiv_drop_rate=None, indiv_fc_units=None, indiv_fc_drop=None,
        indiv_pool_mode='same', indiv_att_dr=0, **g_theta_kwargs):
    person1_joints = []
    persons = []
    extra_objs = []
    if individual:
        for i in range(num_objs):
            object_i = Input(shape=object_shape, name="person1_object"+str(i))
            person1_joints.append(object_i)
    else:
        persons, extra_objs = create_input(num_indivs, num_objs, object_shape, 
               num_extra_objs=num_extra_objs, num_coords_obj=num_coords_obj)
    
    if fuse_type is None:
        g_theta_model = g_theta(object_shape, kernel_init=kernel_init, 
            # drop_rate=drop_rate, model_name="g_theta_"+rel_type, 
            model_name="g_theta_"+rel_type, 
            **g_theta_kwargs)
        #### TODO Insert inter_att_dr here?
        x = create_relationships(rel_type, g_theta_model, 
            person1_joints, [], persons, extra_objs, pool_mode=pool_mode, 
            att_dr=att_dr, rels_dr=rels_dr, out_rels=indiv_out)
    else:
        x = multiple_rels(fuse_type, person1_joints, persons,
                  extra_objs, pool_mode=pool_mode, att_dr=att_dr,
                  object_shape=object_shape, kernel_init=kernel_init, 
                  # drop_rate=drop_rate, 
                  **g_theta_kwargs)
    
    ### indiv_out: creating top
    if indiv_out:
        pooled_rels, rels_outs = x
        x = pooled_rels
        
        indiv_top_kwargs = dict(
            drop_rate=(drop_rate if indiv_drop_rate is None else indiv_drop_rate), 
            fc_units=(fc_units if indiv_fc_units is None else indiv_fc_units), 
            fc_drop=(fc_drop if indiv_fc_drop is None else indiv_fc_drop),
            att_dr=indiv_att_dr)
        
        top_indivs_out = create_top_indiv(pooled_rels, rels_outs, rel_type, 
              kernel_init, indiv_conc_pool=indiv_conc_pool, 
              indiv_pool_mode=indiv_pool_mode, **indiv_top_kwargs)
    
    ### TODO skip creating top, use concat top_indivs_out as out_f_phi?
    
    out_f_phi = create_top(x, kernel_init, fc_units=fc_units, 
                           drop_rate=drop_rate, fc_drop=fc_drop)
    
    
    if individual:
        f_phi_ins = person1_joints
    else:
        f_phi_ins = []
        for p_joints in persons:
            f_phi_ins += p_joints
        for o_coords in extra_objs:
            f_phi_ins += o_coords
    
    if indiv_out:
        out_f_phi = [out_f_phi] + top_indivs_out
    
    model = Model(inputs=f_phi_ins, outputs=out_f_phi, name="f_phi")
    
    return model

def g_theta(object_shape, kernel_init, g_dr=0, g_fc_drop=False, compute_distance=False, 
        compute_motion=False, model_name="g_theta", num_dim=None, overhead=None):
    if compute_motion or compute_distance:
        timesteps = (object_shape[0]-overhead)//num_dim
    def euclideanDistance(inputs):
        if overhead > 0:
            trimmed = [ inputs[0][:,:-overhead], inputs[1][:,:-overhead] ]
        else:
            trimmed = inputs
        coords = [ K.reshape(obj, (-1, timesteps, num_dim) ) for obj in trimmed ] 
        output = K.sqrt(K.sum(K.square(coords[0] - coords[1]), axis=-1))
        return output
    def motionDistance(inputs):
        if overhead > 0:
            trimmed = [ inputs[0][:,:-overhead], inputs[1][:,:-overhead] ]
        else:
            trimmed = inputs
        shifted = [ trimmed[0][:,:-num_dim], trimmed[1][:,num_dim:] ] 
        coords = [ K.reshape(obj, (-1, timesteps-1, num_dim) ) for obj in shifted ]
        output = K.sqrt(K.sum(K.square(coords[0] - coords[1]), axis=-1))
        return output
    
    drop_rate = g_dr
    fc_drop = (g_fc_drop or g_dr>0)
    
    object_i = Input(shape=object_shape, name="object_i")
    object_j = Input(shape=object_shape, name="object_j")
    
    g_theta_inputs = [object_i, object_j]
    if compute_distance:
        distances = Lambda(euclideanDistance, 
            output_shape=lambda inp_shp: (inp_shp[0][0], timesteps),
            name=model_name+'_distanceMerge')([object_i, object_j])
        g_theta_inputs.append(distances)
    
    if compute_motion:
        motions = Lambda(motionDistance, 
            output_shape=lambda inp_shp: (inp_shp[0][0], timesteps-1),
            name=model_name+'_motionMerge')([object_i, object_j])
        g_theta_inputs.append(motions)
        
    x = Concatenate()(g_theta_inputs)
    
    x = Dense(1000, activation='relu', kernel_initializer=kernel_init,
        name=model_name+"_fc1")(x)
    if fc_drop: x = Dropout(drop_rate)(x)
    x = Dense(1000, activation='relu', kernel_initializer=kernel_init,
        name=model_name+"_fc2")(x)
    if fc_drop: x = Dropout(drop_rate)(x)
    x = Dense(1000, activation='relu', kernel_initializer=kernel_init,
        name=model_name+"_fc3")(x)
    if fc_drop: x = Dropout(drop_rate)(x)
    out_g_theta = Dense(500, activation='relu', kernel_initializer=kernel_init,
        name=model_name+"_fc4")(x)
        # name="g_theta_fc4")(x)
    
    model = Model(inputs=[object_i, object_j], outputs=out_g_theta, name=model_name)
    
    return model

def get_fused_model(selected_joints, object_shape, output_size,
        models_kwargs, weights_filepaths, num_dim=None, overhead=None, 
        kernel_init_type='TruncatedNormal', kernel_init_param=0.045, 
        kernel_init_seed=None, drop_rate=0, freeze_g_theta=False, 
        fc_units=None, fc_drop=None, indiv_out=False, grp_out=True,
        indiv_conc_pool=False, indiv_pool_mode='same', indiv_att_dr=0,
        indiv_drop_rate=None, indiv_fc_units=None, indiv_fc_drop=None,
        fuse_at_fc1=False, fuse_at_rels=False, indiv_output_size=0, 
        rels_dr=0, pool_mode='avg', att_dr=0):
    
    prunned_models = []
    models_sel_joints = []
    for model_kwargs, weights_filepath in zip(models_kwargs, weights_filepaths):
        model_kwargs = model_kwargs.copy()
        model_kwargs['num_dim'] = num_dim
        model_kwargs['overhead'] = overhead
        
        model_selected_joints = model_kwargs.pop('selected_joints')
        models_sel_joints.append(model_selected_joints)
        num_joints = len(model_selected_joints)
        
        model = get_model(num_objs=num_joints, object_shape=object_shape, 
            output_size=output_size, indiv_output_size=indiv_output_size, 
            **model_kwargs)
        if weights_filepath is not None and weights_filepath != []:
            model.load_weights(weights_filepath)
        
        if fuse_at_rels:
            # pool_l_names = ('avg_'+model_kwargs.get('rel_type'),)
            # # Ex: avg_indivs_p00, avg_indivs_p01, avg_inter-direct_p00
            # rels_out = []
            # for layer in model.layers:
            #     if layer.name.startswith(pool_l_names):
            #         print(layer.name, layer.output)
            #         rels_out.append(layer.output)
            
            num_indivs = model_kwargs.get('num_indivs', 2)
            layer_name_tpl = 'avg_'+model_kwargs.get('rel_type')+'_p{:0>2}'
            rels_out = []
            for indiv_idx in range(num_indivs):
                layer = model.get_layer(layer_name_tpl.format(indiv_idx))
                rels_out.append(layer.output)
            
            prunned_model = Model(inputs=model.input, outputs=rels_out,
                                  name='irn_'+model_kwargs.get('rel_type'))
        elif not fuse_at_fc1:
            layer = model.get_layer('avg_'+model_kwargs.get('rel_type'))
            out_pool = layer.output
            prunned_model = Model(inputs=model.input, outputs=out_pool,
                                  name='irn_'+model_kwargs.get('rel_type'))
        else: # Prune keeping dropout + f_phi_fc1
            for layer in model.layers[::-1]: # reverse looking for last f_phi_fc1 layer
                if layer.name.startswith(('f_phi_fc1')):
                    out_f_phi_fc1 = layer.output
                    break
            prunned_model = Model(inputs=model.input, outputs=out_f_phi_fc1,
                                  name='irn_'+model_kwargs.get('rel_type'))
        
        if freeze_g_theta:
            for layer in prunned_model.layers: # Freezing model
                layer.trainable = False
        prunned_models.append(prunned_model)
    
    # Train params
    # drop_rate = train_kwargs.get('drop_rate', 0.1)
    # kernel_init_type = train_kwargs.get('kernel_init_type', 'TruncatedNormal')
    # kernel_init_param = train_kwargs.get('kernel_init_param', 0.045)
    # kernel_init_seed = train_kwargs.get('kernel_init_seed')
    
    kernel_init = get_kernel_init(kernel_init_type, param=kernel_init_param, 
        seed=kernel_init_seed)
    
    ## Building Inputs
    models_n_indivs = [ d.get('num_indivs', 2) for d in models_kwargs]
    num_indivs = max(models_n_indivs)
    models_n_x_objs = [d.get('num_extra_objs', 0) for d in models_kwargs]
    num_extra_objs = max(models_n_x_objs)
    models_n_c_objs = [d.get('num_coords_obj', 0) for d in models_kwargs]
    num_coords_obj = max(models_n_c_objs)
    num_joints = len(selected_joints)
    persons, extra_objs = create_input(num_indivs, num_joints, object_shape, 
                  num_extra_objs=num_extra_objs, num_coords_obj=num_coords_obj)
    
    models_outs = []
    for model_idx in range(len(models_kwargs)):
        num_indivs = models_n_indivs[model_idx]
        num_extra_objs = models_n_x_objs[model_idx]
        model_sel_joints = models_sel_joints[model_idx]
        m = prunned_models[model_idx]
        
        model_joints_idx = [ selected_joints.index(joint) 
                            for joint in model_sel_joints ]
        
        model_inputs = []
        for p_joints in persons[:num_indivs]:
            model_inputs += [ p_joints[idx] for idx in model_joints_idx ]
        
        for o_coords in extra_objs[:num_extra_objs]:
            model_inputs += o_coords
        models_outs.append( m(model_inputs) )
    
    if not fuse_at_rels:
        x = Concatenate()(models_outs)
    else:
        p_rels_conc = []
        for p_idx in range(num_indivs):
            p_rels_out = [ out[p_idx] for out in models_outs]
            conc_name ='conc_p{:0>2}'.format(p_idx)
            p_rels_conc.append( Concatenate(name=conc_name)(p_rels_out) )
        
        if rels_dr > 0:
            p_rels_conc = [ Dropout(rels_dr)(rels_out) 
                            for rels_out in p_rels_conc ]
            
        pooled_rels = pool_relations(p_rels_conc, pool_mode, 
                                     name_suffix='fusion', att_dr=att_dr)
        x = pooled_rels
    
    # Building top and Model
    top_kwargs = get_relevant_kwargs(model_kwargs, create_top)
    top_kwargs['drop_rate'] = drop_rate
    if fc_units is not None:
        top_kwargs['fc_units'] = fc_units
    if fc_drop is not None:
        top_kwargs['fc_drop'] = fc_drop
    
    
    x = create_top(x, kernel_init, **top_kwargs)
    
    out_rn = Dense(output_size, activation='softmax', 
        kernel_initializer=kernel_init, name='softmax')(x)
    
    ### grp+indivs
    if indiv_out:
        override_kwargs = remove_None_kwargs(dict(drop_rate=indiv_drop_rate,
                                                  fc_units=indiv_fc_units, 
                                                  fc_drop=indiv_fc_drop,
                                                  att_dr=indiv_att_dr))
        indiv_top_kwargs = top_kwargs.copy()
        indiv_top_kwargs.update(override_kwargs)
        
        top_indivs_out = create_top_indiv(pooled_rels, p_rels_conc, 
                'rels-fusion', kernel_init, indiv_conc_pool=indiv_conc_pool, 
                indiv_pool_mode=indiv_pool_mode, **indiv_top_kwargs)
        
        preds_indivs = Dense(indiv_output_size, activation='softmax', 
            kernel_initializer=kernel_init, name='softmax_indivs')
        
        preds_indivs_out = []
        for indiv_top_out in top_indivs_out:
            preds_indivs_out.append(preds_indivs(indiv_top_out))
            
        if grp_out:
            out_rn = [out_rn] + preds_indivs_out
        else:
            out_rn = preds_indivs_out
    
    inputs = []
    for p_joints in persons:
        inputs += p_joints
    for o_coords in extra_objs:
        inputs += o_coords
    model = Model(inputs=inputs, outputs=out_rn, name="fused_rel_net")
    
    return model

