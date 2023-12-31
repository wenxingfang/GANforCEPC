#!/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python 
import h5py
import numpy as np
#from keras.layers import Input, Lambda, Activation, AveragePooling2D, UpSampling2D, Dense
#from keras.layers.merge import add, concatenate, multiply
#from keras.models import Model
#from keras.layers.merge import multiply
#import keras.backend as K
#K.set_image_dim_ordering('tf')
from keras.models import model_from_json
from keras.models import model_from_yaml
from keras.models import load_model
import tensorflow as tf
import argparse
import sys
#sys.path.append('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/models/')
#from architectures import build_discriminator_3D, sparse_softmax, build_generator_3D, build_regression
#from ops import scale, inpainting_attention, calculate_energy


def get_parser():
    parser = argparse.ArgumentParser(
        description='Run generator. '
        'Sensible defaults come from ...',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--nb-events', action='store', type=int, default=10,
                        help='Number of events to be generatored.')
    parser.add_argument('--latent-size', action='store', type=int, default=512,
                        help='size of random N(0, 1) latent space to sample')
    parser.add_argument('--output', action='store', type=str,
                        help='output file.')
    parser.add_argument('--gen-model-in', action='store',type=str,
                        default='',
                        help='input of gen model')
    parser.add_argument('--gen-weight-in', action='store',type=str,
                        default='',
                        help='input of gen weight')
    parser.add_argument('--comb-model-in', action='store',type=str,
                        default='',
                        help='input of combined model')
    parser.add_argument('--comb-weight-in', action='store',type=str,
                        default='',
                        help='input of combined weight')
    parser.add_argument('--exact-model', action='store',type=bool,
                        default=False,
                        help='use exact input to generate')
    parser.add_argument('--exact-list', action='store',type=str,
                        default='',
                        help='exact event list to generate')

    return parser


if __name__ == '__main__':
    
    parser = get_parser()
    parse_args = parser.parse_args()
    
    gen_out = parse_args.output
    hf = h5py.File(gen_out, 'w')

    ### using generator model  ############
    #print('gen model=',parse_args.gen_model_in)
    #gen_model = model_from_yaml(open(parse_args.gen_model_in).read())
    #print('gen weight=',parse_args.gen_weight_in)
    #gen_model.load_weights(parse_args.gen_weight_in)

    gen_model = load_model(parse_args.gen_model_in, custom_objects={'tf': tf})

    n_gen_images = parse_args.nb_events
    noise            = np.random.normal ( 0 , 1, (n_gen_images, parse_args.latent_size))
    sampled_energies = np.random.uniform(0.1, 1, (n_gen_images, 1))
    sampled_theta    = np.random.uniform(-1 , 1, (n_gen_images, 1))
    sampled_phi      = np.random.uniform(-1 , 1, (n_gen_images, 1))
    sampled_info     = np.concatenate((sampled_theta, sampled_phi, sampled_energies),axis=-1)
    if parse_args.exact_model:
        f_info=open(parse_args.exact_list, 'r')
        index_line=0
        for line in f_info:
            (theta, phi, Energy) = line.split(',')
            theta = (float(theta.split('=')[-1])-90)/45
            phi   = float(phi  .split('=')[-1])/10
            Energy= float(Energy.split('=')[-1])/100
            #print('exact input=', theta, ':', phi, ':', Energy)
            sampled_info[index_line, 0]=theta
            sampled_info[index_line, 1]=phi
            sampled_info[index_line, 2]=Energy
            index_line = index_line + 1
            if index_line >= n_gen_images:
                print('Error: more than nb_events to produce, ignore rest part')
                break
        f_info.close()
    generator_inputs = [noise, sampled_info]
    images = gen_model.predict(generator_inputs, verbose=True)
    #### transfer to real parameters ##############################
    actual_info      = sampled_info.copy()
    actual_info[:,0] = actual_info[:,0]*45 + 90    
    actual_info[:,1] = actual_info[:,1]*10      
    actual_info[:,2] = actual_info[:,2]*100 
    #print ('actual_info\n:',actual_info[0:10])

    hf.create_dataset('Barrel_Hit', data=images)
    hf.create_dataset('MC_info'   , data=actual_info)
    ### using combined model to check discriminator and regression part  ############
    if parse_args.comb_model_in !='':
        comb_model = load_model(parse_args.comb_model_in, custom_objects={'tf': tf})
        results = comb_model.predict(generator_inputs, verbose=True)
        results[1][:,0] = results[1][:,0]*45 + 90
        results[1][:,1] = results[1][:,1]*10
        results[1][:,2] = results[1][:,2]*100
        hf.create_dataset('Disc_fake_real' , data=results[0])
        hf.create_dataset('Reg'            , data=results[1])
    ### save results ############
    hf.close()
    print ('Saved h5 file, done')
