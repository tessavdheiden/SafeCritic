""" Adapted implementation of attention model from: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning """
""" This module was inspired by the paper 'Show, Attend and Tell: Neural Image Caption Generation with Visual Attention' """

import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attention_Encoder(nn.Module):
    """
    Encoder. It encodes the raw scene images with a ResNet pretrained on ImageNet
    """

    def __init__(self, encoded_image_size=14):
        """
        :param encoded_image_size: the width and height of the encoded image [encoded_image_size * encoded_image_size]
        """

        super(Attention_Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet50(pretrained=True)  # pretrained ImageNet ResNet-50

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune(False)

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder (only the
        first layers should retain most of the important information/feature of the images).

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of safeGAN decoder's output
        :param attention_dim: size of the attention network layers that transform the encoded image and the previous
                              SafeGAN hidden state
        :param encoded_image_size: used to add a Batch Normalization layer after the image is encoded (to limit their range of values)
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform SafeGAN decoder's output
        self.relu = nn.ReLU()
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous SafeGAN decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, attention_weights
        """
        image_features = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        hidden_features = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(image_features + hidden_features.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        attention_weights = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * attention_weights.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, attention_weights


class Attention_Decoder(nn.Module):
    """
    Decoder. It uses the Attention weighted image and the previous SafeGAN hidden state to compute the output
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, encoder_dim):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size, dimension of the additional features to attach to the attention weighted image
                          (for example the agents coordinates and their relative displacements with each other)
        :param decoder_dim: size of SafeGAN generator decoder's output
        :param encoder_dim: feature size of encoded images
        :param encoded_image_size: used to tell Attention module if it has to add or not a Batch Normalization
                                   layer to normalize the encoded image
        """
        super(Attention_Decoder, self).__init__()

        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, attention_dim, bias=True)  # decoding LSTM
        self.hidden = self.init_hidden()  # initialize hidden and cell state of the decoding LSTM

    def init_hidden(self):
        """
        :return: two tensors with zeroes to initialize the decoder's LSTM cell state and hidden state
        """
        return (torch.zeros(1, self.attention_dim).to(device),
                torch.zeros(1, self.attention_dim).to(device))

    def forward(self, encoder_out, curr_hidden, embed_info):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param curr_hidden: previous hidden state of SafeGAN generator's decoder, of dimension (batch_size, decoder_dim)
        :param embed_info: additional info to attach to the attention weighted image, for example the agents
                           coordinates and their relative displacements with each other (batch_size, embed_dim)
        :return: Attention output, Attention_weights
        """

        # attention-weighting the encoder's output based on the SafeGAN generator decoder's previous hidden state output
        attention_weighted_encoding, attention_weights = self.attention(encoder_out, curr_hidden)
        lstm_hidden, lstm_cell = self.decode_step(torch.cat([embed_info, attention_weighted_encoding], dim=1),
                                                 (self.hidden[0].repeat(embed_info.shape[0], 1), self.hidden[1].repeat(embed_info.shape[0], 1)))  # (batch_size, attention_dim)
        # update the attention decoder's lstm cell state and hidden state assigning them the first element of the compute
        # output. This because each time the batch size can vary, so it was not possible to have an hidden state with the
        # same dimension of the batch size.
        self.hidden = (lstm_hidden[0], lstm_cell[0])

        return lstm_hidden, attention_weights





""" Temporary code for the visualization of attention weights """

'''import matplotlib.cm as cm
import skimage.transform
from PIL import Image
from datasets.calculate_static_scene_boundaries import get_pixels_from_world'''

'''PLOT ATTENTION WEIGHTS'''
'''if i==0 and self.giuseppe==0:
    plt.clf()
    image = Image.open("/home/q472489/Desktop/FLORA/code/social_gan/datasets/safegan_dataset/SDD/segmented_scenes/"+seq_scenes[i]+".jpg")
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)
    plt.imshow(image)
    h_matrix = pd.read_csv("/home/q472489/Desktop/FLORA/code/social_gan/datasets/safegan_dataset/SDD/"+ seq_scenes[i] + '/{}_homography.txt'.format(seq_scenes[i]), delim_whitespace=True, header=None).values
    original_image_size = Image.open("/home/q472489/Desktop/FLORA/code/social_gan/datasets/safegan_dataset/SDD/"+seq_scenes[i]+"/annotated_boundaries.jpg").size
    pixels = get_pixels_from_world(curr_end_pos, h_matrix, True)
    pixels = pixels*(14*24/original_image_size[0], 14*24/original_image_size[1])
    plt.scatter(pixels[:, 0], pixels[:, 1], marker='.', color="r")
    attention_weights = attention_weights.view(-1, self.encoded_image_size, self.encoded_image_size).detach().cpu().numpy()
    alpha = skimage.transform.pyramid_expand(attention_weights[0], upscale=24, sigma=8)
    plt.imshow(alpha, alpha=0.7)
    plt.set_cmap(cm.Greys_r)
    plt.axis('off')
    plt.show()
    self.giuseppe += 1'''