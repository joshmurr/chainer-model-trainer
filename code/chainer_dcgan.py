#!/usr/bin/env python3
import sys
import os
# import copy
# import six
# import subprocess

from absl import app
from absl import flags
# from absl import logging

import chainer
import chainer.cuda
# from chainer.dataset import concat_examples
# from chainer import function
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer import training
from chainer.training import extension
from chainer.training import extensions
import numpy as np
from PIL import Image


def record_setting(out):
    """Record scripts and commandline arguments"""
    out = out.split()[0].strip()
    if not os.path.exists(out):
        os.system('mkdir -p %s' % out)
        # os.mkdir(out)
    # subprocess.call("cp *.py %s" % out, shell=True)

    with open(out + "/command.txt", "w") as f:
        f.write(" ".join(sys.argv) + "\n")


def sample_generate_light(gen, dst, rows=5, cols=5, seed=0, subdir='preview'):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols
        xp = gen.xp
        z = Variable(xp.asarray(gen.make_hidden(n_images)))
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = gen(z)
        x = chainer.cuda.to_cpu(x.data)
        np.random.seed()

        x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
        _, _, H, W = x.shape
        x = x.reshape((rows, cols, 3, H, W))
        x = x.transpose(0, 3, 1, 4, 2)
        x = x.reshape((rows * H, cols * W, 3))

        preview_dir = '{}/{}'.format(dst, subdir)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)

        preview_path = preview_dir + '/image_latest.png'
        Image.fromarray(x).save(preview_path)
        preview_path = preview_dir + '/image{:0>8}.png'.format(trainer.updater.iteration)
        Image.fromarray(x).save(preview_path)

    return make_image


def sample_generate(gen, dst, rows=10, cols=10, seed=0, subdir='preview'):
    """Visualization of rows*cols images randomly generated by the generator."""

    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols
        xp = gen.xp
        z = Variable(xp.asarray(gen.make_hidden(n_images)))
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = gen(z)
        x = chainer.cuda.to_cpu(x.data)
        np.random.seed()

        x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
        _, _, h, w = x.shape
        x = x.reshape((rows, cols, 3, h, w))
        x = x.transpose(0, 3, 1, 4, 2)
        x = x.reshape((rows * h, cols * w, 3))

        preview_dir = '{}/{}'.format(dst, subdir)
        preview_path = preview_dir + '/image{:0>8}.png'.format(trainer.updater.iteration)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        Image.fromarray(x).save(preview_path)

    return make_image


class DCGANGenerator64(chainer.Chain):
    def __init__(self,
                 n_hidden=128,
                 bottom_width=4,
                 ch=512,
                 wscale=0.02,
                 z_distribution="normal",
                 hidden_activation=F.relu,
                 output_activation=F.tanh,
                 use_bn=True):
        super().__init__()
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_width = bottom_width
        self.z_distribution = z_distribution
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.use_bn = use_bn

        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0 = L.Linear(self.n_hidden, bottom_width * bottom_width * ch, initialW=w)
            self.dc1 = L.Deconvolution2D(ch, ch // 2, 4, 2, 1, initialW=w)
            self.dc2 = L.Deconvolution2D(ch // 2, ch // 4, 4, 2, 1, initialW=w)
            self.dc3 = L.Deconvolution2D(ch // 4, ch // 8, 4, 2, 1, initialW=w)
            self.dc4 = L.Deconvolution2D(ch // 8, 3, 4, 2, 1, initialW=w)
            if self.use_bn:
                self.bn0 = L.BatchNormalization(bottom_width * bottom_width * ch)
                self.bn1 = L.BatchNormalization(ch // 2)
                self.bn2 = L.BatchNormalization(ch // 4)
                self.bn3 = L.BatchNormalization(ch // 8)

    def make_hidden(self, batchsize):
        if self.z_distribution == "normal":
            return np.random.randn(batchsize, self.n_hidden, 1, 1) \
                .astype(np.float32)
        elif self.z_distribution == "uniform":
            return np.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)) \
                .astype(np.float32)
        else:
            raise Exception("unknown z distribution: %s" % self.z_distribution)

    def __call__(self, z):
        if not self.use_bn:
            h = F.reshape(self.hidden_activation(self.l0(z)), (len(z), self.ch, self.bottom_width, self.bottom_width))
            h = self.hidden_activation(self.dc1(h))
            h = self.hidden_activation(self.dc2(h))
            h = self.hidden_activation(self.dc3(h))
            x = self.output_activation(self.dc4(h))
        else:
            h = F.reshape(
                self.hidden_activation(self.bn0(self.l0(z))), (len(z), self.ch, self.bottom_width, self.bottom_width))
            h = self.hidden_activation(self.bn1(self.dc1(h)))
            h = self.hidden_activation(self.bn2(self.dc2(h)))
            h = self.hidden_activation(self.bn3(self.dc3(h)))
            x = self.output_activation(self.dc4(h))
        return x


class DCGANDiscriminator64(chainer.Chain):
    def __init__(self, bottom_width=4, ch=512, wscale=0.02, output_dim=1):
        w = chainer.initializers.Normal(wscale)
        super().__init__()
        with self.init_scope():
            self.c0_0 = L.Convolution2D(3, ch // 8, 4, 2, 1, initialW=w)
            self.c0_1 = L.Convolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w)
            self.c1_0 = L.Convolution2D(ch // 4, ch // 4, 3, 1, 1, initialW=w)
            self.c1_1 = L.Convolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w)
            self.c2_0 = L.Convolution2D(ch // 2, ch // 2, 3, 1, 1, initialW=w)
            self.c2_1 = L.Convolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w)
            self.c3_0 = L.Convolution2D(ch // 1, ch // 1, 3, 1, 1, initialW=w)
            self.l4 = L.Linear(bottom_width * bottom_width * ch, output_dim, initialW=w)
            self.bn0_1 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn1_0 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn1_1 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn2_0 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn2_1 = L.BatchNormalization(ch // 1, use_gamma=False)
            self.bn3_0 = L.BatchNormalization(ch // 1, use_gamma=False)

    def __call__(self, x):
        h = F.leaky_relu(self.c0_0(x))
        h = F.leaky_relu(self.bn0_1(self.c0_1(h)))
        h = F.leaky_relu(self.bn1_0(self.c1_0(h)))
        h = F.leaky_relu(self.bn1_1(self.c1_1(h)))
        h = F.leaky_relu(self.bn2_0(self.c2_0(h)))
        h = F.leaky_relu(self.bn2_1(self.c2_1(h)))
        h = F.leaky_relu(self.bn3_0(self.c3_0(h)))
        return self.l4(h)

# ******* GAN 128 x 128 ************************************************ #

class DCGANGenerator128(chainer.Chain):
    def __init__(self,
                 n_hidden=128,
                 bottom_width=4,
                 ch=1024,
                 wscale=0.02,
                 z_distribution="normal",
                 hidden_activation=F.relu,
                 output_activation=F.tanh,
                 use_bn=True):
        super().__init__()
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_width = bottom_width
        self.z_distribution = z_distribution
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.use_bn = use_bn

        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0 = L.Linear(self.n_hidden, bottom_width * bottom_width * ch, initialW=w)
            self.dc1 = L.Deconvolution2D(ch, ch // 2, 4, 2, 1, initialW=w)
            self.dc2 = L.Deconvolution2D(ch // 2, ch // 4, 4, 2, 1, initialW=w)
            self.dc3 = L.Deconvolution2D(ch // 4, ch // 8, 4, 2, 1, initialW=w)
            self.dc4 = L.Deconvolution2D(ch // 8, ch // 16, 4, 2, 1, initialW=w)
            self.dc5 = L.Deconvolution2D(ch // 16, 3, 4, 2, 1, initialW=w)
            if self.use_bn:
                self.bn0 = L.BatchNormalization(bottom_width * bottom_width * ch)
                self.bn1 = L.BatchNormalization(ch // 2)
                self.bn2 = L.BatchNormalization(ch // 4)
                self.bn3 = L.BatchNormalization(ch // 8)
                self.bn4 = L.BatchNormalization(ch // 16)

    def make_hidden(self, batchsize):
        if self.z_distribution == "normal":
            return np.random.randn(batchsize, self.n_hidden, 1, 1) \
                .astype(np.float32)
        elif self.z_distribution == "uniform":
            return np.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)) \
                .astype(np.float32)
        else:
            raise Exception("unknown z distribution: %s" % self.z_distribution)

    def __call__(self, z):
        if not self.use_bn:
            h = F.reshape(self.hidden_activation(self.l0(z)), (len(z), self.ch, self.bottom_width, self.bottom_width))
            h = self.hidden_activation(self.dc1(h))
            h = self.hidden_activation(self.dc2(h))
            h = self.hidden_activation(self.dc3(h))
            h = self.output_activation(self.dc4(h))
            x = self.output_activation(self.dc5(h))
        else:
            h = F.reshape(
                self.hidden_activation(self.bn0(self.l0(z))), (len(z), self.ch, self.bottom_width, self.bottom_width))
            h = self.hidden_activation(self.bn1(self.dc1(h)))
            h = self.hidden_activation(self.bn2(self.dc2(h)))
            h = self.hidden_activation(self.bn3(self.dc3(h)))
            x = self.output_activation(self.dc4(h))
        return x


class DCGANDiscriminator128(chainer.Chain):
    def __init__(self, bottom_width=4, ch=1024, wscale=0.02, output_dim=1):
        w = chainer.initializers.Normal(wscale)
        super().__init__()
        with self.init_scope():
            self.c0_0 = L.Convolution2D(3, ch // 16, 4, 2, 1, initialW=w)
            self.c0_1 = L.Convolution2D(ch // 16, ch // 8, 4, 2, 1, initialW=w)
            self.c1_0 = L.Convolution2D(ch // 8, ch // 8, 4, 2, 1, initialW=w)
            self.c1_1 = L.Convolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w)
            self.c2_0 = L.Convolution2D(ch // 4, ch // 4, 3, 1, 1, initialW=w)
            self.c2_1 = L.Convolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w)
            self.c3_0 = L.Convolution2D(ch // 2, ch // 2, 3, 1, 1, initialW=w)
            self.c3_1 = L.Convolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w)
            self.c4_0 = L.Convolution2D(ch // 1, ch // 1, 3, 1, 1, initialW=w)
            self.l4 = L.Linear(bottom_width * bottom_width * ch, output_dim, initialW=w)

            self.bn0_1 = L.BatchNormalization(ch // 8, use_gamma=False)
            self.bn1_0 = L.BatchNormalization(ch // 8, use_gamma=False)
            self.bn1_1 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn2_0 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn2_1 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn3_0 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn3_1 = L.BatchNormalization(ch // 1, use_gamma=False)
            self.bn4_0 = L.BatchNormalization(ch // 1, use_gamma=False)

    def __call__(self, x):
        h = F.leaky_relu(self.c0_0(x))
        h = F.leaky_relu(self.bn0_1(self.c0_1(h)))
        h = F.leaky_relu(self.bn1_0(self.c1_0(h)))
        h = F.leaky_relu(self.bn1_1(self.c1_1(h)))
        h = F.leaky_relu(self.bn2_0(self.c2_0(h)))
        h = F.leaky_relu(self.bn2_1(self.c2_1(h)))
        h = F.leaky_relu(self.bn3_0(self.c3_0(h)))
        h = F.leaky_relu(self.bn3_1(self.c3_1(h)))
        h = F.leaky_relu(self.bn4_0(self.c4_0(h)))
        return self.l4(h)

# ******* GAN 128 x 128 ************************************************ #

class ResNetResBlockUp(chainer.Chain):
    def __init__(self, in_ch, out_ch=None, wscale=0.02):
        super().__init__()
        out_ch = out_ch or in_ch
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.c0 = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(out_ch, out_ch, 3, 1, 1, initialW=w)
            self.cs = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)
            self.bn0 = L.BatchNormalization(in_ch)
            self.bn1 = L.BatchNormalization(out_ch)

    def __call__(self, x):
        h = self.c0(F.unpooling_2d(F.relu(self.bn0(x)), 2, 2, 0, cover_all=False))
        h = self.c1(F.relu(self.bn1(h)))
        hs = self.cs(F.unpooling_2d(x, 2, 2, 0, cover_all=False))
        return h + hs


class ResNetResBlockDown(chainer.Chain):
    def __init__(self, in_ch, out_ch=None, wscale=0.02):
        super().__init__()
        out_ch = out_ch or in_ch
        self.in_ch = in_ch
        self.out_ch = out_ch

        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.c0 = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(out_ch, out_ch, 4, 2, 1, initialW=w)
            self.cs = L.Convolution2D(in_ch, out_ch, 4, 2, 1, initialW=w)
            self.bn0 = L.BatchNormalization(in_ch)
            self.bn1 = L.BatchNormalization(out_ch)

    def __call__(self, x):
        self.h0 = x
        self.h1 = self.c0(F.relu(self.h0))
        self.h2 = self.c1(F.relu(self.h1))
        self.h3 = self.cs(self.h0)
        self.h4 = self.h2 + self.h3
        return self.h4


class LinkRelu(chainer.Chain):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return F.relu(x)


class LinkTanh(chainer.Chain):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return F.tanh(x)


class ResNetInputDense(chainer.Chain):
    def __init__(self, n_hidden, bottom_width, ch, wscale=0.02):
        super().__init__()
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_width = bottom_width
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l = L.Linear(self.n_hidden, bottom_width * bottom_width * ch, initialW=w)

    def __call__(self, z):
        return F.reshape(self.l(z), (len(z), self.ch, self.bottom_width, self.bottom_width))


class ResNetOutputDense(chainer.Chain):
    def __init__(self, bottom_width, ch, n_output, wscale=0.02):
        super().__init__()
        self.ch = ch
        self.bottom_width = bottom_width
        self.n_output = n_output
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l = L.Linear(bottom_width * bottom_width * ch, self.n_output, initialW=w)

    def __call__(self, z):
        z = F.reshape(z, (len(z), self.ch * self.bottom_width * self.bottom_width))
        return self.l(z)


class ResNetGenerator128(chainer.Chain):
    def __init__(self, n_hidden=128, bottom_width=4, ch=1024, wscale=0.02, z_distribution="normal"):
        super().__init__()
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_width = bottom_width
        self.z_distribution = z_distribution

        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.dense = ResNetInputDense(n_hidden, bottom_width, ch)
            self.resblockups = chainer.ChainList(
                ResNetResBlockUp(ch, ch),
                ResNetResBlockUp(ch, ch // 2),
                ResNetResBlockUp(ch // 2, ch // 4),
                ResNetResBlockUp(ch // 4, ch // 8),
                ResNetResBlockUp(ch // 8, ch // 16),
            )
            self.finals = chainer.ChainList(
                L.BatchNormalization(ch // 16),
                LinkRelu(),
                L.Convolution2D(ch // 16, 3, 3, 1, 1, initialW=w),
                LinkTanh(),
            )

    def make_hidden(self, batchsize):
        if self.z_distribution == "normal":
            return np.random.randn(batchsize, self.n_hidden, 1, 1) \
                .astype(np.float32)
        elif self.z_distribution == "uniform":
            return np.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)) \
                .astype(np.float32)
        else:
            raise Exception("unknown z distribution: %s" % self.z_distribution)

    def __call__(self, x):
        h = x
        h = self.dense(h)
        for _layers in self.resblockups:
            h = _layers(h)
        for _layer in self.finals:
            h = _layer(h)
        return h


class ResNetDiscriminator128(chainer.Chain):
    def __init__(self, bottom_width=4, ch=1024, wscale=0.02, output_dim=1):
        super().__init__()
        self.bottom_width = bottom_width
        self.ch = ch
        self.wscale = wscale
        self.output_dim = output_dim

        with self.init_scope():
            self.resblockdowns = chainer.ChainList(
                ResNetResBlockDown(3, ch // 16),
                ResNetResBlockDown(ch // 16, ch // 8),
                ResNetResBlockDown(ch // 8, ch // 4),
                ResNetResBlockDown(ch // 4, ch // 2),
                ResNetResBlockDown(ch // 2, ch),
            )
            self.finals = chainer.ChainList(LinkRelu(), ResNetOutputDense(bottom_width, ch, output_dim))

    def __call__(self, x):
        h = x
        for _layers in self.resblockdowns:
            h = _layers(h)
        for _layer in self.finals:
            h = _layer(h)
        return h


class ResNetGenerator256(chainer.Chain):
    def __init__(self, n_hidden=128, bottom_width=4, ch=1024, wscale=0.02, z_distribution="normal"):
        super().__init__()
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_width = bottom_width
        self.z_distribution = z_distribution

        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.dense = ResNetInputDense(n_hidden, bottom_width, ch)
            self.resblockups = chainer.ChainList(
                ResNetResBlockUp(ch, ch),
                ResNetResBlockUp(ch, ch // 2),
                ResNetResBlockUp(ch // 2, ch // 4),
                ResNetResBlockUp(ch // 4, ch // 8),
                ResNetResBlockUp(ch // 8, ch // 16),
                ResNetResBlockUp(ch // 16, ch // 32),
            )
            self.finals = chainer.ChainList(
                L.BatchNormalization(ch // 32),
                LinkRelu(),
                L.Convolution2D(ch // 32, 3, 3, 1, 1, initialW=w),
                LinkTanh(),
            )

    def make_hidden(self, batchsize):
        if self.z_distribution == "normal":
            return np.random.randn(batchsize, self.n_hidden, 1, 1) \
                .astype(np.float32)
        elif self.z_distribution == "uniform":
            return np.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)) \
                .astype(np.float32)
        else:
            raise Exception("unknown z distribution: %s" % self.z_distribution)

    def __call__(self, x):
        h = x
        h = self.dense(h)
        for _layers in self.resblockups:
            h = _layers(h)
        for _layer in self.finals:
            h = _layer(h)
        return h


class ResNetDiscriminator256(chainer.Chain):
    def __init__(self, bottom_width=4, ch=1024, wscale=0.02, output_dim=1):
        super().__init__()
        self.bottom_width = bottom_width
        self.ch = ch
        self.wscale = wscale
        self.output_dim = output_dim

        with self.init_scope():
            self.resblockdowns = chainer.ChainList(
                ResNetResBlockDown(3, ch // 32),
                ResNetResBlockDown(ch // 32, ch // 16),
                ResNetResBlockDown(ch // 16, ch // 8),
                ResNetResBlockDown(ch // 8, ch // 4),
                ResNetResBlockDown(ch // 4, ch // 2),
                ResNetResBlockDown(ch // 2, ch),
            )
            self.finals = chainer.ChainList(LinkRelu(), ResNetOutputDense(bottom_width, ch, output_dim))

    def __call__(self, x):
        h = x
        for _layers in self.resblockdowns:
            h = _layers(h)
        for _layer in self.finals:
            h = _layer(h)
        return h


def dcgan_loss_real(y):
    return F.sum(F.softplus(-y)) / np.prod(y.shape)


def dcgan_loss_fake(y):
    return F.sum(F.softplus(y)) / np.prod(y.shape)


def loss_l2(h, t):
    return F.sum((h - t)**2) / np.prod(h.data.shape)


def copy_param(target_link, source_link):
    """Copy parameters of a link to another link."""
    target_params = dict(target_link.namedparams())
    for param_name, param in source_link.namedparams():
        target_params[param_name].data[:] = param.data

    # Copy Batch Normalization's statistics
    target_links = dict(target_link.namedlinks())
    for link_name, link in source_link.namedlinks():
        if isinstance(link, L.BatchNormalization):
            target_bn = target_links[link_name]
            target_bn.avg_mean[:] = link.avg_mean
            target_bn.avg_var[:] = link.avg_var


def soft_copy_param(target_link, source_link, tau):
    """Soft-copy parameters of a link to another link."""
    target_params = dict(target_link.namedparams())
    for param_name, param in source_link.namedparams():
        target_params[param_name].data[:] *= (1 - tau)
        target_params[param_name].data[:] += tau * param.data

    # Soft-copy Batch Normalization's statistics
    target_links = dict(target_link.namedlinks())
    for link_name, link in source_link.namedlinks():
        if isinstance(link, L.BatchNormalization):
            target_bn = target_links[link_name]
            target_bn.avg_mean[:] *= (1 - tau)
            target_bn.avg_mean[:] += tau * link.avg_mean
            target_bn.avg_var[:] *= (1 - tau)
            target_bn.avg_var[:] += tau * link.avg_var


class DRAGANUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis, self.smoothed_gen = kwargs.pop('models')
        self.lambda_gp = kwargs.pop('lambda_gp')
        self.smoothing = kwargs.pop('smoothing')
        self.learning_rate = kwargs.pop('learning_rate')
        self.learning_rate_anneal = kwargs.pop('learning_rate_anneal')
        self.learning_rate_anneal_trigger = kwargs.pop('learning_rate_anneal_trigger')
        self.learning_rate_anneal_interval = kwargs.pop('learning_rate_anneal_interval')
        super().__init__(*args, **kwargs)

    def get_x_real_data(self, batch, batch_size):
        xp = self.gen.xp
        x_real_data = []
        for i in range(batch_size):
            this_instance = batch[i]
            if isinstance(this_instance, tuple):
                this_instance = this_instance[0]  # It's (data, data_id), so take the first one.
            x_real_data.append(np.asarray(this_instance).astype("f"))
        x_real_data = xp.asarray(x_real_data)
        return x_real_data

    def get_z_fake_data(self, batch_size):
        xp = self.gen.xp
        return xp.asarray(self.gen.make_hidden(batch_size))

    def update_core(self):
        xp = self.gen.xp

        opt_g = self.get_optimizer('gen')
        opt_d = self.get_optimizer('dis')

        # z: latent | x: data | y: dis output
        # *_real/*_fake/*_pertubed: Variable
        # *_data: just data (xp array)

        batch = self.get_iterator('main').next()
        batch_size = len(batch)

        x_real_data = self.get_x_real_data(batch, batch_size)
        z_fake_data = self.get_z_fake_data(batch_size)

        x_real = Variable(x_real_data)
        z_fake = Variable(z_fake_data)

        x_fake = self.gen(z_fake)
        y_fake = self.dis(x_fake)
        loss_gen = dcgan_loss_real(y_fake)
        chainer.report({'loss_adv': loss_gen}, self.gen)

        self.gen.cleargrads()
        loss_gen.backward()
        opt_g.update()
        x_fake.unchain_backward()

        # keep smoothed generator.
        soft_copy_param(self.smoothed_gen, self.gen, 1.0 - self.smoothing)

        # alternative gradient update
        x_fake = self.gen(z_fake)
        x_fake.unchain_backward()
        y_fake = self.dis(x_fake)
        if self.lambda_gp > 0:
            y_real = self.dis(x_real)
            loss_adv = dcgan_loss_real(y_real) + dcgan_loss_fake(y_fake)
            '''
            # WGAN-GP specific start
            eta = xp.random.uniform(
                0, 1, size=batch_size).astype("f")[:, None, None, None]
            x_perturbed = Variable(
                (x_fake.data * eta + (1.0 - eta) * x_real.data).astype('f'))
            # WGAN-GP specific ends
            '''
            # DRAGAN specific starts
            std_x_real_data = xp.std(x_real.data, axis=0, keepdims=True)
            rnd_x = xp.random.uniform(-1, 1, x_real.data.shape).astype("f")
            x_perturbed = Variable((x_real.data + 0.5 * rnd_x * std_x_real_data).astype('f'))
            # DRAGAN specific ends

            y_perturbed = self.dis(x_perturbed)

            grad_x_perturbed, = chainer.grad([y_perturbed], [x_perturbed], enable_double_backprop=True)
            grad_l2 = F.sqrt(F.sum(grad_x_perturbed**2, axis=(1, 2, 3)))
            loss_gp = self.lambda_gp * loss_l2(grad_l2, 1.0)

            loss_dis = loss_adv + loss_gp

            chainer.report({'loss_adv': loss_adv, 'loss_gp': loss_gp}, self.dis)
        else:
            y_real = self.dis(x_real)
            loss_adv = dcgan_loss_real(y_real) + dcgan_loss_fake(y_fake)

            loss_dis = loss_adv

            chainer.report({'loss_adv': loss_adv}, self.dis)

        self.dis.cleargrads()
        loss_dis.backward()
        opt_d.update()

        if (self.learning_rate_anneal > 0 and self.iteration >= self.learning_rate_anneal_trigger
                and self.iteration % self.learning_rate_anneal_interval == 0):
            self.update_learning_rate()

    def update_learning_rate(self):
        opt_g = self.get_optimizer('gen')
        opt_d = self.get_optimizer('dis')
        iter = self.iteration - self.learning_rate_anneal_trigger
        if iter >= 0:
            iter = iter // self.learning_rate_anneal_interval

            opt_g.alpha = self.learning_rate * (self.learning_rate_anneal**iter)
            opt_d.alpha = self.learning_rate * (self.learning_rate_anneal**iter)
            print('anneled lr for g:', opt_g.alpha)
            print('anneled lr for d:', opt_d.alpha)


FLAGS = flags.FLAGS

# algorithm & architecture
flags.DEFINE_string('arch', '', 'Architecture of netowrk. can be `dcgan64` or `resnet128` or `resnet256`.')
flags.DEFINE_integer('image_size', 32, 'Size of image.')

# hps (training dynamics)
# flags.DEFINE_integer('seed', 19260817, '')
flags.DEFINE_integer('batch_size', 64, '')
flags.DEFINE_float('adam_alpha', 0.0002, 'alpha in Adam optimizer')
flags.DEFINE_float('adam_beta1', 0.5, 'beta1 in Adam optimizer')
flags.DEFINE_float('adam_beta2', 0.999, 'beta2 in Adam optimizer')
flags.DEFINE_integer('max_iter', 100000, '')
flags.DEFINE_float('lambda_gp', 1.0, 'Lambda for gradient panelty.')
flags.DEFINE_float('smoothing', 0.999, '')
flags.DEFINE_float('learning_rate_anneal', 0.0, 'anneal the learning rate. 0 for no annealing.')
flags.DEFINE_integer('learning_rate_anneal_trigger', 20000, 'trigger of learning rate anneal')
flags.DEFINE_integer('learning_rate_anneal_interval', 10000, 'interval of learning rate anneal')

# hps (device)
flags.DEFINE_integer('gpu', 0, 'GPU ID (negative value indicates CPU)')

# hps (I/O)
flags.DEFINE_string('npz_path', '', 'path to dataset npz file')
flags.DEFINE_string('out', 'result', 'Directory to output the result')
flags.DEFINE_integer('snapshot_interval', 10000, 'Interval of snapshot')
flags.DEFINE_integer('evaluation_interval', 10000, 'Interval of heavy evaluation')
flags.DEFINE_integer('evaluation_sample_interval', 500, 'Interval of evaluation sampling')
flags.DEFINE_integer('display_interval', 100, 'Interval of displaying log to console')


def make_optimizer(model, alpha, beta1, beta2):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
    optimizer.setup(model)
    return optimizer


def main(argv):
    del argv  # Unused.

    record_setting(FLAGS.out)
    report_keys = ['epoch', 'iteration', 'elapsed_time']

    device = FLAGS.gpu

    # Set up dataset and its iterator
    X_train = np.load(FLAGS.npz_path)['size_%d' % FLAGS.image_size]
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    train_dataset = X_train

    train_iter = chainer.iterators.SerialIterator(train_dataset, FLAGS.batch_size)

    # Setup algorithm specific networks and updaters
    models = []
    opts = {}
    updater_args = {
        "iterator": {
            'main': train_iter
        },
        "device": device,
        'lambda_gp': FLAGS.lambda_gp,
        'smoothing': FLAGS.smoothing,
        'learning_rate': FLAGS.adam_alpha,
        'learning_rate_anneal': FLAGS.learning_rate_anneal,
        'learning_rate_anneal_trigger': FLAGS.learning_rate_anneal_trigger,
        'learning_rate_anneal_interval': FLAGS.learning_rate_anneal_interval,
    }

    Updater = DRAGANUpdater

    if FLAGS.arch == 'dcgan64':
        generator_class = DCGANGenerator64
        discriminator_class = DCGANDiscriminator64
        assert FLAGS.image_size == 64
    elif FLAGS.arch == 'dcgan128':
        generator_class = DCGANGenerator128
        discriminator_class = DCGANDiscriminator128
        assert FLAGS.image_size == 128
    elif FLAGS.arch == 'resnet128':
        generator_class = ResNetGenerator128
        discriminator_class = ResNetDiscriminator128
        assert FLAGS.image_size == 128
    elif FLAGS.arch == 'resnet256':
        generator_class = ResNetGenerator256
        discriminator_class = ResNetDiscriminator256
        assert FLAGS.image_size == 256
    else:
        raise ValueError('Unknown -arch %s' % FLAGS.arch)

    generator = generator_class()
    discriminator = discriminator_class()
    smoothed_generator = generator_class()
    models = [generator, discriminator, smoothed_generator]
    model_names = ['Generator', 'Discriminator', 'SmoothedGenerator']
    report_keys.extend(["gen/loss_adv", "dis/loss_adv", 'dis/loss_gp'])
    updater_args['lambda_gp'] = FLAGS.lambda_gp

    if device > -1:
        chainer.cuda.get_device_from_id(device).use()
        print("use gpu {}".format(device))
        for model in models:
            model.to_gpu()

    # Set up optimizers
    opts["gen"] = make_optimizer(generator, FLAGS.adam_alpha, FLAGS.adam_beta1, FLAGS.adam_beta2)
    opts["dis"] = make_optimizer(discriminator, FLAGS.adam_alpha, FLAGS.adam_beta1, FLAGS.adam_beta2)

    updater_args["optimizer"] = opts
    updater_args["models"] = models

    # Set up updater and trainer
    updater = Updater(**updater_args)
    trainer = training.Trainer(updater, (FLAGS.max_iter, 'iteration'), out=FLAGS.out)

    # Set up extensions
    for model, model_name in zip(models, model_names):
        trainer.extend(
            extensions.snapshot_object(model, model_name + '_{.updater.iteration}.npz'),
            trigger=(FLAGS.snapshot_interval, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(
        sample_generate_light(generator, FLAGS.out),
        trigger=(FLAGS.evaluation_sample_interval, 'iteration'),
        priority=extension.PRIORITY_WRITER)
    trainer.extend(
        sample_generate_light(smoothed_generator, FLAGS.out, rows=4, cols=4, subdir='preview_smoothed'),
        trigger=(FLAGS.evaluation_sample_interval, 'iteration'),
        priority=extension.PRIORITY_WRITER)
    trainer.extend(extensions.LogReport(keys=report_keys, trigger=(FLAGS.display_interval * 5, 'iteration')))
    trainer.extend(extensions.PrintReport(report_keys), trigger=(FLAGS.display_interval, 'iteration'))

    # Run the training
    trainer.run()


import pdb, traceback, sys # code  # noqa

if __name__ == '__main__':
    try:
        app.run(main)
    except Exception:  # noqa
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
