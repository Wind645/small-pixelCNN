# small-pixelCNN
I was reading the article published recently by Kaiming He, where it discusses about a new paradigm for generative model.
The key idea is to connect small units of generative model in a fractal way, and the original unit model took autoregressive 
model.

PixelCNN is the very kind of autoregressive model that can generate pictures, but the small demo I implemented here is not 
yet the proper model, for it samples pictures directly from p(x) without receiving an input, that goes against the idea of the 
unit model presented in the article. But take it easy, as you can see, I didn't even load the data here for training, it's a small
practice for me so that I can understand the concrete architecture of an autoregressive model. I will correct those details later on. 

I will try to reproduce a toy model of the fractal model in Kaiming He's paper, and here is simply a small component that I 
have already finished. 

I wish that I can make it...
