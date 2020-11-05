#!/usr/bin/env python
# coding: utf-8

# file:///home/hamidi/Downloads/2020-06-21-kr_extended-SHL2020%20(1).png![image.png](attachment:image.png)

# In[1]:


import tensorflow as tf
import numpy as np


# In[2]:


from dataset import DataReader
from refinedsample import RefinedSample

train = DataReader(what='train')
validation = DataReader(what='validation')


# In[3]:


import stjohns as ag


# ## Select desired position-dependent circuit within the base model

# In[7]:


print("########################################")
print("trainVsSubway")
print("fineTune_1 on Hips")
print("fineTune_2 on Hand")
print("fineTune_3 on Torso")
print("fineTune_4 on Bag")
print("fineTune_5 on Bag: increase learning rate")
print("########################################")
print("WalkRunBike")
print("fineTune_6 on Bag:")
print("fineTune_7 on Hips:")
print("########################################")
print("fineTune_8 on Hand:")
print("+ reduce the learning rate (after being increased previously)")
print("########################################")
print("fineTune_9/Hand, fineTune_9.gpu_out:")
print("+ Hand +++ stillVsAll")
print("+ 3 Dense layers rather than 2 on top")
print("########################################")
print("fineTune_9/Hips, fineTune_10.gpu_out:")
print("+ Hips +++ stillVsAll")
print("+ 2 Dense layers rather than 3 on top")
print("=> process stopped @epoch 30 @fine-tuning step")
print("########################################")
print("fineTune_9/Bag, fineTune_11.gpu_out:")
print("+ Bag +++ stillVsAll")
print("=> process stopped @epoch 30 @fine-tuning step")
print("########################################")
print("fineTune_9/Torso, fineTune_12.gpu_out:")
print("+ Torso +++ stillVsAll")
print("+ reduced number of training epochs to 29")
print("=> process stopped @epoch 29 @fine-tuning step")
print("########################################")
print("fineTune_FeetWheelsRail/Hand, fineTune_13.gpu_out:")
print("+ Hand +++ FeetWheelsRail")
print("########################################")
print("fineTune_WalkRunBike/Hand, fineTune_14.gpu_out:")
print("+ Hand +++ WalkRunBike")
print("########################################")
print("fineTune_carVsBus/Hand, fineTune_15.gpu_out:")
print("+ Hand +++ carVsBus")
print("########################################")
print("fineTune_trainVsSubway/Hand, fineTune_16.gpu_out:")
print("+ Hand +++ trainVsSubway")
print("########################################")
print("fineTune_stillVsAll/Hand, fineTune_17.gpu_out:")
print("+ Hand +++ stillVsAll")

position = 'Hand'

#callbacks
#checkpoint_path = "{}/{}/cp.ckpt".format(nni.get_experiment_id(), nni.get_sequence_id())
save_checkpoint_path = "fineTune_stillVsAll/"+position+"/cp.{epoch:04d}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=save_checkpoint_path,
    save_weights_only=True,
    verbose=1,
    save_freq='epoch')



##########################################################################################
# # Fine-tuning trainVsSubway model

# In[4]:


# stillVsAll model (stjohns_41)
checkpoint_path = 'stjohns_41'
epoch = 45
#map original labels to the new learning problem
new_classes = {
        0: ['still'],
        1: ['walk', 'run', 'bike', 'car', 'bus', 'train', 'subway']
}
num_classes = len(list(new_classes.keys()))
refinedSampleId = 'stillVsAll'


# FeetWheelsRail model (stjohns_38)
# checkpoint_path = 'stjohns_38'
# epoch = 50
# #map original labels to the new learning problem
# new_classes = {
#         0: ['walk', 'run', 'bike'],
#         1: ['car', 'bus'],
#         2: ['train', 'subway']
# }
# num_classes = len(list(new_classes.keys()))
# refinedSampleId = 'FeetWheelsRail'


# WalkRunBike model (stjohns_40)
# checkpoint_path = 'stjohns_40'
# epoch = 46
# #map original labels to the new learning problem
# new_classes = {
#         0: ['walk'],
#         1: ['run'],
#         2: ['bike']
# }
# num_classes = len(list(new_classes.keys()))
# refinedSampleId = 'WalkRunBike'


# carVsBus model (stjohns_42)
# checkpoint_path = 'stjohns_42'
# epoch = 48
# #map original labels to the new learning problem
# new_classes = {
#         0: ['car'],
#         1: ['bus']
# }
# num_classes = len(list(new_classes.keys()))
# refinedSampleId = 'carVsBus'


# trainVsSubway model (stjohns_43)
# checkpoint_path = 'stjohns_43'
# epoch = 50
# #map original labels to the new learning problem
# new_classes = {
#     0: ['train'],
#     1: ['subway']
# }
# num_classes = len(list(new_classes.keys()))
# refinedSampleId = 'trainVsSubway'
##########################################################################################




# load data
train_dict, train_labels = ag.load_data(what='train', new_classes=new_classes, refinedSampleId=refinedSampleId)
valid_dict, valid_labels = ag.load_data(what='validation', new_classes=new_classes, refinedSampleId=refinedSampleId)


# In[5]:


train_dict['Hips_Pressure'][34]


# ## Base model

# In[6]:


#Base model
model = ag.stjohns(ag.dict_of_params(ag.params), num_classes=num_classes)
model.load_weights(checkpoint_path+'/cp.{epoch:04d}.ckpt'.format(epoch=epoch))


# In[8]:


inputs = [input for input in model.inputs if input.name.startswith(position)]
view = model.get_layer('view_'+position).output
base_model = tf.keras.Model(inputs=inputs, outputs=view)
#base_model.load_weights(sg.checkpoint_path.format(epoch=1))

#Freeze model for now
base_model.trainable = False  #freeze base_model's weights
assert base_model.get_layer('view_'+position).trainable == False   # check if freezed (eg. dense layer view_Torso)


# ## Top additional layers

# In[11]:


#Top additional layers
view = base_model(inputs=inputs, training=False)  # <--- not to confuse with trainable. Here it means that the base model is used solely in inference mode

view = tf.keras.layers.Dense(40, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                bias_regularizer=tf.keras.regularizers.l2(0.001),
                activity_regularizer=tf.keras.regularizers.l2(0.001))(view)
view = tf.keras.layers.Dense(30, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                bias_regularizer=tf.keras.regularizers.l2(0.001),
                activity_regularizer=tf.keras.regularizers.l2(0.001))(view)
class_outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(view)

model = tf.keras.Model(inputs=inputs, outputs=class_outputs)
# model.summary()
#tf.keras.utils.plot_model(model, 'washington_fine_tune.png', show_shapes=True)


# In[12]:


#train the top layers

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),  # , amsgrad=True),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['acc'])

# history = model.fit(
model.fit(
    train_dict,
    tf.keras.utils.to_categorical(train_labels, num_classes=num_classes),
    batch_size=512,
    epochs=2,
    # callbacks=[ReportIntermediates(), cp_callback],
    # validation_split=0.3,
    validation_data=(valid_dict, tf.keras.utils.to_categorical(valid_labels, num_classes=num_classes)),
    shuffle=True
)

print('top layers trained successfully')

# print(history.history['acc'])
# print(history.history['val_acc'])



# ## Fine-tune

# In[ ]:


# Fine-tune
# Unfreeze the base_model. Note that it keeps running in inference mode
# since we passed `training=False` when calling it. This means that
# the batchnorm layers will not update their batch statistics.
# This prevents the batchnorm layers from undoing all the training
# we've done so far.
base_model.trainable = True
# model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # , amsgrad=True),  # <--- lower learning rate (according to documentation)
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['acc'])

history = model.fit(
    train_dict,
    tf.keras.utils.to_categorical(train_labels, num_classes=num_classes),
    batch_size=512,
    epochs=29,
    # callbacks=[ReportIntermediates(), cp_callback],
    callbacks=[cp_callback],
    # validation_split=0.3,
    validation_data=(valid_dict, tf.keras.utils.to_categorical(valid_labels, num_classes=num_classes)),
    shuffle=True
)


# ## Check predictions

# In[ ]:


print(history.history['acc'])
print(history.history['val_acc'])


