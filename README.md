# Style-Transfer-via-Multi-Scale-Feature-CNN
Applying a Style Transfer Objective by solely using a CNN model (without any Discriminant or Adversarial network)!  
  
Style transfer is a computer vision technique which takes two images – a content image and a style reference image, and blends them together so that the resulting output image retains the core elements of the content image, but yet appears to be as "painted" in the style of the style reference image.  
  
Style transfer, as a task performed in CV world, has obtained much attention in the DL field.  
There are several ways in which to create a style transfer outcome. Such include but not limited to, Variational Auto-Encoders (VAE), Generative Adversarial Networks (GANs), Transformers (Encoder-Decoder with Attention), Neural style, and more deep independent architectures.  
  
Our goal in this project is to be able to create a style transferred image depending solely on a some known DL neural architecture without relying on other network model which functions as a Discriminator objective, or just another capacity model for dividing the content and style tasks.  
  
I believe if the style transfer task were thought of as a combination of DC (or base-line hidden distribution) component from the content image (could be thought of as the dc of the image in the fourier domain, plus a certain Heaviside filter for part of the low frequencies of the content – Low Pass FIR Filter). Hence, maintaining Semantic regions.  
And an AC distribution part – as for a variant (variance) component which potentially holds Semantic information, object shape, and Semantic Texture of the style image (could also be thought of as for the high frequency part in the fourier domain – as for a High Pass FIR filter).  
Then, the downstream task of such would be feasible!  
  
There has been research papers on the matter (Pioneer: LA Gatys et al. 2016). And here's another.  
  
A few notes,  
* The algorithm, architecture, and inner transformations works amazingly!  
* The code for this project was built as an independent entity, and can be run via Terminal with UI command line.  
  The interface offers plenty of flexibility and automation.  
  It has its own interface, with default values and helper functions with a built-in manual on how to operate.  
  
  
In this folder you can find the following,  
1. The core Style Transfer via MSF CNN code  
2. A detailed, well explained, file for the whole process,  
  2.1. Algorithm  
  2.2. Architecture of the model  
  2.3. Illustrations  
  2.4. Mathematical proofs to each step taken whic supports the implemented method  
3. A further presentation for better visualization!  
4. A detailed explanation on how to run the code  
5. Content and Style images used together with its results  
  
  
  
That's it.  
Have fun with the code, and try exploring out of the features!  
Enjoy! :)  
  
