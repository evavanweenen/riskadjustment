import numpy as np
import pandas as pd

import keras
from keras.layers import Input, Dense, Embedding, Concatenate, Dropout, dot
from keras.models import Model
from keras.initializers import Constant
from keras import backend as K

from config import datadir

class REGR:
    def __init__(self, model_type, diag_shape, patient_shape, hosp_shape, cohort_shape, num_hosp, mean_y, lambda_1=1e-6, lambda_2=1e-6):
        self.diag_shape = diag_shape
        self.patient_shape = patient_shape
        self.hosp_shape = hosp_shape
        self.cohort_shape = cohort_shape
        
        self.num_hosp = num_hosp
        
        self.mean_y = mean_y

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        
        self.nodes_out = 1
        self.act_out = 'sigmoid'
        self.optimizer = 'Adam'
        self.loss = 'binary_crossentropy'
        self.bias_init = Constant(value=np.log(self.mean_y / (1.-self.mean_y)))
        
        if model_type == 'hglm':
            self.model = self.hglm_model()
        elif model_type == 'lasso':
            self.model = self.lasso_model()
        else:
            print('Model type unknown')
    
    def l1_l2_reg(self, W):
        return self.lambda_1 * K.sum(K.abs(W[0:self.num_hosp,:])) + self.lambda_2 * K.sum(K.square(W[0:self.num_hosp,:]))
    
    def l2_reg(self, W):
        return self.lambda_2 * K.sum(K.square(W[0:self.num_hosp,:]))

    def hglm_model(self):
        # input layer
        input_diag = Input(shape = (self.diag_shape,))
        input_patient = Input(shape = (self.patient_shape,))
        input_hosp = Input(shape = (self.hosp_shape,))
        input_cohort = Input(shape = (self.cohort_shape,))
        
        # concatenate patients and diagnosis
        out = Concatenate()([input_patient, input_diag])        

        # one layer for each cohort
        c1 = Dense(1, name = 'cohort1', activation = 'linear', use_bias = False)(out)
        c2 = Dense(1, name = 'cohort2', activation = 'linear', use_bias = False)(out)
        c3 = Dense(1, name = 'cohort3', activation = 'linear', use_bias = False)(out)
        c4 = Dense(1, name = 'cohort4', activation = 'linear', use_bias = False)(out)
        c5 = Dense(1, name = 'cohort5', activation = 'linear', use_bias = False)(out)
        
        #
        cohort_effect = Concatenate()([c1, c2, c3, c4, c5])
        dot_product = dot([input_cohort, cohort_effect], axes=1, normalize=False)
        
        out = Concatenate()([input_hosp, dot_product])
        out = Dense(self.nodes_out, activation = self.act_out, 
                    bias_initializer = self.bias_init,
                    kernel_regularizer =  self.l2_reg,
                    name = "output")(out)
        
        self.model = Model(inputs=[input_diag, input_patient, input_hosp, input_cohort], outputs = out)
        self.model.compile(optimizer = self.optimizer, loss = self.loss)
        
        return self.model 
        
    def lasso_model(self):
        # input layer
        input_diag = Input(shape = (self.diag_shape,))
        input_patient = Input(shape = (self.patient_shape,))
        input_hosp = Input(shape = (self.hosp_shape,))
        input_cohort = Input(shape = (self.cohort_shape,)) #this is not used
        
        # concatenate patients and diagnosis
        out = Concatenate()([input_patient, input_diag])

        # one layer for each cohort
        c1 = Dense(1, name = 'cohort1', activation = 'linear', use_bias = False)(out)
        c2 = Dense(1, name = 'cohort2', activation = 'linear', use_bias = False)(out)
        c3 = Dense(1, name = 'cohort3', activation = 'linear', use_bias = False)(out)
        c4 = Dense(1, name = 'cohort4', activation = 'linear', use_bias = False)(out)
        c5 = Dense(1, name = 'cohort5', activation = 'linear', use_bias = False)(out)
        
        #
        cohort_effect = Concatenate()([c1, c2, c3, c4, c5])
        dot_product = dot([input_cohort, cohort_effect], axes=1, normalize=False)
  
        out = Concatenate()([input_hosp, dot_product])
        out = Dense(self.nodes_out, activation = self.act_out, 
                    bias_initializer = self.bias_init,
                    kernel_regularizer =  self.l1_l2_reg,
                    name = "output")(out)
        
        self.model = Model(inputs=[input_diag, input_patient, input_hosp, input_cohort], outputs = out)
        self.model.compile(optimizer = self.optimizer, loss = self.loss)
        
        return self.model


class NN:
    def __init__(self, model_type, diag_shape, patient_shape, hosp_shape, num_hosp, mean_y, layers_diag = 3, layers_patient = 2, nodes_diag = 1024, nodes_patient = 512, dropout = .5, act_diag = 'relu', act_patient = 'relu', shrinkage = 0.00001):
        # inputs        
        self.diag_shape = diag_shape            # number of diagnoses for one patient
        self.patient_shape = patient_shape      # patient features
        self.hosp_shape = hosp_shape            # number of hospitals
        self.num_hosp = num_hosp                

        self.mean_y = mean_y

        self.embeddings = pd.read_csv(datadir+'embeddings/icd2emb.csv', header = None).set_index(0).values

        self.model_type = model_type # which of the models underneath to use
        self.model = self.create_model(layers_diag, layers_patient, nodes_diag, nodes_patient, dropout, act_diag, act_patient, shrinkage)

    def create_model(self, layers_diag, layers_patient, nodes_diag, nodes_patient, dropout, act_diag, act_patient, shrinkage):
        """
        Initialize hyperparameters. This function can be used as an sklearn wrapper around a keras model for gridsearch.
        Sklearn wrapper around keras model for gridsearch requires the arguments to not be anything of list-type, therefore inefficient long list of inputs and cannot be self.bladiebla
        """
        # hidden layers
        self.layers_diag = layers_diag ; self.layers_patient = layers_patient
        
        # hidden nodes
        self.nodes_diag = nodes_diag ; self.nodes_patient = nodes_patient
        self.nodes_out = 1
        
        # activation functions
        self.act_diag = act_diag ; self.act_patient = act_patient
        self.act_out = 'sigmoid'
        
        # dropout
        self.dropout = dropout
        
        # optimization
        self.optimizer = 'Adam'
        self.loss = 'binary_crossentropy'
        self.bias_init = Constant(value=np.log(self.mean_y/(1.-self.mean_y)))
        self.shrinkage = shrinkage
        
        # model type
        if self.model_type == 'nn':
            return self.set_model()
        elif self.model_type == 'black_box':
            return self.black_box()
        elif self.model_type == 'hospital_only':
            return self.hospital_only()
        else:
            print('Model type unknown')

    def l2_reg(self, W):
        return self.shrinkage * K.sum(K.square(W[0:self.num_hosp,:]))
    
    def set_model(self): 
        num_embeddings, embedding_dim = self.embeddings.shape

        #input layer
        input_diag = Input(shape = (self.diag_shape,))
        input_patient = Input(shape = (self.patient_shape,))
        input_hosp = Input(shape = (self.hosp_shape,))

        #------------------------------------- embeddings layer -------------------------------------#
        # embeddings layer
        diag_emb = Embedding(num_embeddings,
                             embedding_dim,
                             weights=[self.embeddings],
                             input_length=self.diag_shape,
                             trainable=True)(input_diag)

        #-------------------------------------- deepset layer --------------------------------------#
        # select primary disease
        diag_primary_emb = Lambda(lambda x: x[:, 0, :], name = "Lambda_" + str(0))(diag_emb)
        
        # take sum, minimum and maximum of 8 icd code embeddings (vectors) of one patient
        Adder = Lambda(lambda x: K.sum(x, axis=1), output_shape=(lambda shape: (shape[0], shape[2])))
        Maxer = Lambda(lambda x: K.max(x, axis=1), output_shape=(lambda shape: (shape[0], shape[2])))
        Miner = Lambda(lambda x: K.min(x, axis=1), output_shape=(lambda shape: (shape[0], shape[2])))

        mined = Miner(diag_emb)
        maxed = Maxer(diag_emb)
        added = Adder(diag_emb)

        # concatenate (merge) outputs of maxed, added, mined and the primary diagnosis embedding
        out = Concatenate()([maxed, added, mined, diag_primary_emb])

        # diagnosis
        for i in range(self.layers_diag-1):
            out = Dense(self.nodes_diag, activation = self.act_diag)(out)
            out = Dropout(self.dropout)(out)
        deepset = Dense(self.nodes_diag, activation = self.act_diag)(out) #no dropout in last layer

        # add patients
        dnn = Concatenate()([input_patient, deepset])
        for i in range(self.layers_patient):
            dnn = Dense(self.nodes_patient, activation = self.act_patient)(dnn)
            dnn = Dropout(self.dropout)(dnn)
        dnn = Dense(1, activation = 'linear', use_bias = False)(dnn)

        #--------------------------------- hospital-specific effect ---------------------------------#
        # add hospital
        out = Concatenate()([input_hosp, dnn])

        # output layer
        out = Dense(self.nodes_out, activation = self.act_out,
                    bias_initializer = self.bias_init,
                    kernel_regularizer = self.l2_reg,
                    name = "output")(out)
                    

        self.model = Model(inputs=[input_diag, input_patient, input_hosp], outputs = out)
        self.model.compile(optimizer = self.optimizer, loss = self.loss)

        return self.model 
        
    def black_box(self): 
        num_embeddings, embedding_dim = self.embeddings.shape

        # input layer
        input_diag = Input(shape = (self.diag_shape,))
        input_patient = Input(shape = (self.patient_shape,))
        input_hosp = Input(shape = (self.hosp_shape,))

        #------------------------------------- embeddings layer -------------------------------------#
        # embeddings layer
        diag_emb = Embedding(num_embeddings,
                             embedding_dim,
                             weights=[self.embeddings],
                             input_length=self.diag_shape,
                             trainable=True)(input_diag)

        #-------------------------------------- deepset layer --------------------------------------#
        # select primary disease
        diag_primary_emb = Lambda(lambda x: x[:, 0, :], name = "Lambda_" + str(0))(diag_emb)
        
        # take sum, minimum and maximum of 8 icd code embeddings (vectors) of one patient
        Adder = Lambda(lambda x: K.sum(x, axis=1), output_shape=(lambda shape: (shape[0], shape[2])))
        Maxer = Lambda(lambda x: K.max(x, axis=1), output_shape=(lambda shape: (shape[0], shape[2])))
        Miner = Lambda(lambda x: K.min(x, axis=1), output_shape=(lambda shape: (shape[0], shape[2])))

        mined = Miner(diag_emb)
        maxed = Maxer(diag_emb)
        added = Adder(diag_emb)

        # concatenate (merge) outputs of maxed, added, mined and the primary diagnosis embedding
        out = Concatenate()([maxed, added, mined, diag_primary_emb])

        # diagnosis
        for i in range(self.layers_diag-1):
            out = Dense(self.nodes_diag, activation = self.act_diag)(out)
            out = Dropout(self.dropout)(out)
        deepset = Dense(self.nodes_diag, activation = self.act_diag)(out) #no dropout in last layer

        # add patients and hospital
        dnn = Concatenate()([input_hosp, input_patient, deepset])
        for i in range(self.layers_patient):
            dnn = Dense(self.nodes_patient, activation = self.act_patient)(dnn)
            dnn = Dropout(self.dropout)(dnn)

        #--------------------------------- hospital-specific effect ---------------------------------#
        # output layer
        out = Dense(self.nodes_out, activation = self.act_out,
                    bias_initializer = self.bias_init,
                    kernel_regularizer = self.l2_reg,
                    name = "output")(dnn)
                    

        self.model = Model(inputs=[input_diag, input_patient, input_hosp], outputs = out)
        self.model.compile(optimizer = self.optimizer, loss = self.loss)

        return self.model 
        
    def hospital_only(self):
        """
        Use only the effect of the hospitals to model the readmission probability
        Don't include comorbidities and patient-specifc effects
        """
        # input layer
        input_diag = Input(shape = (self.diag_shape,)) #not used
        input_patient = Input(shape = (self.patient_shape,)) #not used
        input_hosp = Input(shape = (self.hosp_shape,))

        # output layer
        out = Dense(self.nodes_out, activation = self.act_out,
                    bias_initializer = Constant(value=np.log(self.mean_y / (1.-self.mean_y))),
                    kernel_regularizer = self.l2_reg,
                    name = "output")(input_hosp)
        
        self.model = Model(inputs=[input_diag, input_patient, input_hosp], outputs = out)
        self.model.compile(optimizer = self.optimizer, loss = self.loss)
        
        return self.model 