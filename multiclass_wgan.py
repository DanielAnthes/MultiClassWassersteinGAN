#%%
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from tensorflow.python.profiler import profiler_v2 as profiler

from load_data import load_data_by_class
from modules import make_generator_model, make_discriminator_model

tf.config.run_functions_eagerly(False)
tf.random.set_seed(42)
for d in tf.config.list_physical_devices():
    try:
        tf.config.experimental.set_memory_growth(d, True)
    except:
        print(f"COULD NOT ENABLE MEMORY GROWTH ON DEVICE {d}")

#%%

BATCHSIZE = 32
EPOCHS = 200
NCRIT_UPDATES = 1
SAMPLEFREQ = 1
GP_WEIGHT = 10.
CT_WEIGHT = 2.
LIPSCHITZ_CONST = 1.
CLASSES = list(range(10))
ALPHA=0.1
BETA = 0.1

generators = [make_generator_model() for _ in CLASSES]
critic = make_discriminator_model(len(CLASSES))
optim = tf.keras.optimizers.RMSprop()

ds_train, ds_val = load_data_by_class(BATCHSIZE)


#%%

def wasserstein_loss_gen(ygen):
    return - tf.reduce_mean(ygen)

def wasserstein_loss_critic(ygen, yreal):
    return tf.reduce_mean(ygen) - tf.reduce_mean(yreal)

def wasserstein_loss_critic_multiclass(ygen, yreal, class_idx, nclasses):
    ygen_i = tf.gather(ygen, class_idx, axis=1)
    yreal_i = yreal[:, class_idx]
    loss_i = wasserstein_loss_critic(ygen_i, yreal_i)
    # loss_not_i = ALPHA * (nclasses - 1)

    # for j in range(nclasses):
    #     loss_not_i += tf.maximum(0., loss_i - wasserstein_loss_critic(ygen_i, tf.gather(yreal, j, axis=1)))
    
    return loss_i
    return BETA * loss_i + (BETA / (nclasses-1)) * loss_not_i

def wasserstein_loss_gen_multiclass(ygen, class_idx):
    return wasserstein_loss_gen(tf.gather(ygen, class_idx, axis=1))

def train_step_generator_multiclass(generator, class_idx):
    if not tf.executing_eagerly():
        print("> TRACING GENERATOR UPDATE")
    # generate fake samples
    noise = tf.random.normal([BATCHSIZE, 100])
    trainable_vars = generator.trainable_variables

    with tf.GradientTape() as tape:
        d_fake = generator(noise)
        crit_score_fake = critic(d_fake)
        loss = wasserstein_loss_gen_multiclass(crit_score_fake, class_idx)
    
    grads = tape.gradient(loss, trainable_vars)
    optim.apply_gradients(zip(grads, trainable_vars))
    return loss

def train_step_critic_gp_perturbed_multiclass(d_real, nsamples, generator, class_idx):
    if not tf.executing_eagerly():
        print("> TRACING CRITIC UPDATE")
    # generate fake samples
    noise = tf.random.normal([nsamples, 100])
    d_fake = generator(noise)
    eps = tf.random.uniform((nsamples,1,1,1))
    xhat = eps * d_fake + (1 - eps) * d_real

    trainable_vars = critic.trainable_variables

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(xhat)
        crit_score_fake = critic(d_fake, training=False) 
        crit_score_real = critic(d_real, training=False)
        crit_score_xhat = critic(xhat, training=False)
        crit_score_xprime = critic(d_real, training=True)  # pertrub once
        critic_score_xprimeprime = critic(d_real, training=True)  # perturb twice
        ws_loss = wasserstein_loss_critic_multiclass(crit_score_fake, crit_score_real, class_idx, len(CLASSES))
        grad_xhat = tape.gradient(tf.gather(crit_score_xhat, class_idx, axis=1), xhat)
        grad_penalty = GP_WEIGHT * tf.reduce_mean((tf.norm(tf.reshape(grad_xhat, (nsamples, -1)), axis=1) - LIPSCHITZ_CONST)**2)
        ct_penalty = CT_WEIGHT * tf.reduce_mean(tf.maximum(0., tf.sqrt(tf.reduce_sum((tf.gather(crit_score_xprime, class_idx, axis=1) \
                                                                 - tf.gather(critic_score_xprimeprime, class_idx, axis=1)) ** 2))))  # TODO double check ct penalty
        loss = ws_loss + grad_penalty + ct_penalty

    grads = tape.gradient(loss, trainable_vars)
    optim.apply_gradients(zip(grads, trainable_vars))
    del tape  # clean up persistent tape
    return loss, ws_loss, grad_penalty, ct_penalty


#%%
# keep track of number of critic updates
counter = 0

for i in range(len(CLASSES)):
    generators[i](tf.random.normal((1, 100)))
critic(tf.random.normal((1, 28, 28, 1)))

generators[0].summary()
critic.summary()

# set up data structure for update steps
generator_steps = dict()
critic_steps = dict()

metrics = {
    'critic_loss': tf.keras.metrics.Mean(name="critic_loss"),
    'generator_loss': tf.keras.metrics.Mean(name="generator_loss"),
    'wasserstein_critic': tf.keras.metrics.Mean(name='wasserstein_critic'),
    'gradient_penalty': tf.keras.metrics.Mean(name='gradient_penalty'),
    'ct_penalty': tf.keras.metrics.Mean(name='ct_penalty')
}

train_log_dir = 'logs/gradient_tape/' + 'wgan_multi_simple' + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

critic_counter = 0
generator_counter = 0

for i in range(EPOCHS):
    print(f"EPOCH {i+1}/{EPOCHS}")
    if i % SAMPLEFREQ == 0:
        for c in range(len(CLASSES)):
            noise = tf.random.normal((5, 100))
            samples = generators[c](noise).numpy()

            epoch_str = str(i).zfill(5)
            fig = plt.figure()
            for pnum, s in enumerate(samples):
                plt.subplot(5,1,pnum+1)
                plt.imshow(s.squeeze())
            fig.savefig(f"./samples_multiclass/samples_epoch_{epoch_str}_class_{c}")
            plt.close()


    for batch, data in enumerate(tqdm(ds_train)):
        d_real, c = data
        counter += 1
        c = c.numpy()
        # train critic
        if c not in critic_steps.keys():
            print(c, type(c))
            # compile train step
            print(f"... COMPILING CRITIC UPDATE FOR CLASS {c}")
            critic_steps[c] = tf.function(train_step_critic_gp_perturbed_multiclass)
        critic_loss, ws_loss, grad_penalty, ct_penalty = critic_steps[c](d_real, len(d_real), generators[c], c)
        critic_counter += 1
        metrics["critic_loss"].update_state(critic_loss)
        metrics["wasserstein_critic"].update_state(ws_loss)
        metrics["gradient_penalty"].update_state(grad_penalty)
        metrics["ct_penalty"].update_state(ct_penalty)
        with train_summary_writer.as_default():
            tf.summary.scalar('critic_loss', metrics['critic_loss'].result(), step=critic_counter)
            tf.summary.scalar('wasserstein_critic', metrics['wasserstein_critic'].result(), step=critic_counter)
            tf.summary.scalar('gradient_penalty', metrics['gradient_penalty'].result(), step=critic_counter)
            tf.summary.scalar('ct_penalty', metrics['ct_penalty'].result(), step=critic_counter)


        if counter % NCRIT_UPDATES == 0:
            if c not in generator_steps.keys():
                print(f"... COMPILING GENERATOR UPDATE FOR CLASS {c}")
                generator_steps[c] = tf.function(train_step_generator_multiclass)
            generator_loss = generator_steps[c](generators[c], c)
            generator_counter += 1
            metrics["generator_loss"].update_state(generator_loss)
            with train_summary_writer.as_default():
                tf.summary.scalar('generator_loss', metrics['generator_loss'].result(), step=generator_counter)
            counter = 0
        
    for m in metrics.keys():
        print(f"{m}: {metrics[m].result()}")
        metrics[m].reset_states()

# %%
