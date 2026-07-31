* General rules for writing  
  * If you need to cite something, just put \~\\cite{} and don’t add anything inside the brackets  
  * Don’t edit the .bib file unless explicitly told so  
  * Below there will be sections titled “side information for writing”, which will include useful information for you to understand details  
  * For bullet points titled “figures” below, make a placeholder .tex file for each described figure inside paper\_source/figures/. I will fill out the .tex file with actual figures later. Write tentative captions inside each .tex file as well.  
  * For bullet points titled “tables” below, make a .tex file containing the table that is being described with empty cells. I will fill out the .tex file with actual numbers later  
  * If you want add to anything that is completely new from the outline, mark it in red color  
* Title  
  * Tactile Analogies: Example-Driven Synthesis of Tactile Videos  
* Key Contributions  
  * New Task and Benchmark  
  * Simple and efficient solution  
  * Outperform baselines  
  * Applications in 3D surface reconstruction  
* Abstract  
  *   
* Introduction  
  *   
* Related Work  
  * Tactile Perception  
    * Due to developments in tactile sensors, tactile perception has rapidly developed in the recent years to complement traditional vision-based perception  
    * Various tasks have been tackled using tactile perception, which the aim of aiding challenging scenarios where vision alone might be insufficient   
    * Object/texture recognition  
      * Focus on cases where fine-grained surface details are important for recognition  
      * List of papers, but just cite and don’t explain in detail  
        * Texture classification on uneven surfaces using deep learning techniques  
        * Object Recognition Using Glove Tactile Sensor  
        * Deep learning-assisted object recognition with hybrid triboelectric-capacitive tactile sensor  
        * Object Identification with Tactile Sensors using Bag-of-Features  
        * DOT-Sim: Differentiable Optical Tactile Simulation with Precise Real-to-Sim Physical Calibration  
    * 3D reconstruction  
      * Address cases when vision-based 3D reconstruction fails, e.g. in scenarios when large occlusions are present due to gripper-object interactions or geometry acquisition is difficult due to glossy surface geometry  
      * Tactile sensor for handling occlusions during 3D reconstruction  
        * NeuralFeels with neural fields: Visuotactile perception for in-hand manipulation  
        * TouchSDF: A DeepSDF Approach for 3D Shape Reconstruction Using Vision-Based Tactile Sensing  
      * Tactile sensor for handling glossy surfaces  
        * Snap-it, Tap-it, Splat-it: Tactile-Informed 3D Gaussian Splatting for Reconstructing Challenging Surfaces  
        * SplatTouch: Explicit 3D Representation Binding Vision and Touch  
        * Touch-GS: Visual-Tactile Supervised 3D Gaussian Splatting  
      * In addition, tactile sensors can provide fine-grained surface information, which could lead to detailed 3D reconstruction  
        * GelSLAM: A Real-time, High-Fidelity, and Robust 3D Tactile SLAM System  
    * Touch localization  
      *   
        * TacLoc: Global Tactile Localization on Objects from a Registration Perspective  
      *   
      *   
    * Our paper tackles a new task of predicting tactile videos, which could help improve the performance of various tactile prediction tasks such as object recognition or 3D reconstruction by augmenting the already collected touch samples.  
  * Predicting Sensory Signals from Examples  
    * Vision  
      * A large number of prior works exist  
      * Image analogies  
        *   
    * Audio  
      * Conditional generation of audio from video via foley analogies  
        *   
      *   
    * Touch  
      *   
    * We extend the last strand of works into predicting tactile videos, which is more challenging as we should be able to model the gel dynamics of the tactile sensor   
* Method  
  *   
* Experiments  
  * Experimental Setup  
    * Implementation Details  
      * Key hyperparameter details for retrieval and refinement network training  
      * Training is performed on a single RTX 3090 GPU  
        * Number of epochs trained  
        * How long training takes  
    * Benchmark  
      * We create a benchmark for evaluating tactile analogies using the ObjectFolder dataset  
      * For each of the benchmark below, we construct a dataset that contains for each object in ObjectFolder  
        * Set of tactile normal videos containing the gel normal changes over time  
          * Use tactile simulator Taxim  
        * View-space RGB, heightmaps, normals, and curvature maps rendered at the tactile sensor pose at 1x, 2x, 4x scales   
      * Two types of benchmarks  
        * Ground-truth retrieval benchmark  
          * Pair of tactile videos \+ normal renderings at the tactile sensor pose  
            * One video is used for query, and another is used for reference  
          * Evaluate coarse alignment \+ refinement performance  
          * Side information for writing  
            * Script for building dataset  
              * train\_refine\_scripts/gen\_contact\_query\_tactile\_normal\_pseudo\_mini/  
              * train\_refine\_scripts/gen\_contact\_ref\_tactile\_normal\_pseudo\_mini/  
            * Dataset location  
              * Taxim/results/{gen\_contact\_full\_query\_tactile\_normal\_pseudo\_mini,gen\_contact\_full\_tactile\_normal\_pseudo\_mini}  
        * Full pipeline benchmark  
          * Evaluate full pipeline including retrieval  
          * Collect K touches per object  
          * Full pipeline is evaluated in the following fashion  
            * Leave M touch out and treat them as query  
            * Perform retrieval against the remaining touch locations to pick the reference touch  
            * Perform refinement using pre-trained refinement network  
          * Side information for writing  
            * Script for building benchmark  
              * train\_refine\_scripts/gen\_contact\_raw\_eval\_tactile\_normal\_pseudo\_mini/  
            * Dataset location  
              * Taxim/results/gen\_contact\_raw\_eval\_tactile\_normal\_pseudo\_mini/  
    * Baselines  
      * Since we tackle a new task, we adapt various techniques developed from prior works in tactile perception for our task  
      * Tactile-Augmented Radiance Fields (TaRF)  
        * Train a diffusion model that takes RGB \+ depth maps as input, and outputs tactile images  
        * We train a new version of TaRF using the train split from our ground-truth retrieval benchmark  
          * The tactile frame at the middle of each tactile video is used as the prediction target  
        * Since the prediction is not a video, we tile the predicted frame to a video of same length to the reference  
      * Tactile Normal Quilting  
        * Baseline adapted from Tactile DreamFusion ([https://arxiv.org/abs/2412.06785](https://arxiv.org/abs/2412.06785))  
        * Uses image quilting to tile the tactile normal from the reference touch to the entire object mesh surface, renders the view-space normal at the query tactile sensor pose  
        * After this, integrates the normals into height, and renders a tactile video using Taxim   
      * Implicit Neural Representations  
        * Baseline adapted from ObjectFolder ([https://proceedings.mlr.press/v164/gao22a.html](https://proceedings.mlr.press/v164/gao22a.html))  
        * We train a NeRF-like implicit neural representation that receives pixel coordinates and pressing depths as input and outputs the tactile normal  
        * The implicit neural representation is trained per-sample  
          * The network is trained on the reference touches, and evaluated at query locations  
      * Side information for writing  
        * All baselines are inside baselines/  
  * Performance Comparison  
    * Ground-truth retrieval  
      * Quantitative Results  
        * Tables  
          * Performance comparison of our method against baselines  
            * Include metrics both for coarse transfer and refined transfer using neural network   
      * Qualitative Results  
        * Figures  
          * A two-column figure that shows the prediction results of our method on various touch locations  
            * The figure itself will be a matrix of shape 3 x 6  
              * 1st column: reference touch mid frame  
              * 2nd column: reference normal mid frame  
              * 3rd column: query normal mid frame  
              * 4th column: query touch coarse transfer  
              * 5th column: query touch refined transfer using network  
              * 6th column: ground-truth query touch normal  
      * TODO: finish writing once we have results  
    * Full pipeline  
      * Quantitative Results  
        * Tables  
          * Performance comparison of our method against baselines  
            * Include metrics both for coarse transfer and refined transfer using neural network   
      * Qualitative Results  
        * Figures  
          * A two-column figure that compares the prediction results of our method against the baselines  
          * The figure will be a matrix  
            * Each row will be one method including ours  
              * The first row will show the reference tactile normal video  
            * Each column will be video frames predicted from the tested methods  
              * For image-only predictions, we will put the image on the first frame and write N/A (image only) on the rest of the frames  
      * TODO: finish writing once we have results  
  * Ablation Study  
    * We ablate key components of our method to demonstrate their importance in attaining accurate tactile transfer  
    * Test on a 20 object subset of the full pipeline benchmark  
    * Report results about the following components  
      * Other modalities for retrieval and feature matching during initial alignment (scale fixed to 4x the tactile sensor)  
        * Normals  
        * RGB  
        * Curvature  
        * Height map  
      * Scale used for retrieval and feature matching  
        * 1x, 2x, 4x  
      * w/o Neural network-based refinement  
      * w/o Temporal FiLM  
      * w/o Normal concatenation in network-based refinement  
    * Figures  
      * A small one-column figure that compares the neural network predictions from after various ablations  
        * w/o Neural network-based refinement  
        * w/o Temporal FiLM  
        * w/o Normal concatenation in network-based refinement  
    * Tables  
      * A table summarizing the ablation experiment above  
        * Report PSNR, SSIM, and LPIPS  
    * TODO: finish writing once we have results  
  * Application in 3D Surface Reconstruction and Visuo-tactile Sensor Simulation  
    * Figures  
      * A two-column figure that shows the 3D reconstruction results of our method  
      * The first row will show the reference tactile normal video  
      * The second row will show predicted tactile normal video after neural network inference  
      * The third row will show the heightmap 3D point clouds per frame   
      * The fourth row will show the RGB visuo-tactile video frames  
      * Each column will be the video frames  
    * Using the tactile normals estimated from our method, we can perform 3D reconstruction using Poisson integration and obtain heightmaps  
    * The reconstruction results can be used for embodied agents to reason about fine-grained shape properties of previously untouched locations, or for haptics simulation  
    * Further, by combining the estimated heightmaps with Taxim’s RGB simulation, we can obtain virtual measurements of visuo-tactile sensors  
    * This could potentially aid tactile-based manipulation policies that have been trained directly using visuo-tactile sensor measurements  
      * To illustrate, an embodied agent could simulate a grasp policy by predicting the tactile videos of object contact points before actual physical contact  
  * Runtime Analysis  
    * We report the runtime of our full pipeline. The runtime is measured using a RTX 3090 GPU.  
    * Given N reference touches and 1 query, the time it takes to produce a touch prediction is as follows  
      * Retrieval phase after DINOv3 feature extraction takes Xs  
      * Coarse alignment after local feature matching takes Xs  
      * Neural network-based refinement takes Xs per frame  
    * Due to the short runtime, our method is lightweight and amenable for runtime-critical applications such as AR/VR and robotics.   
* Conclusion  
  * In conclusion, we proposed tactile analogies, an example-driven pipeline for tactile video synthesis  
  * Experiments show that our method outperforms all the tested baselines in terms of video prediction accuracy  
  * We also demonstrate that our predicted tactile normals can be used for 3D surface reconstruction and simulating visuo-tactile measurements without direct physical contact.  
  * We project our work to serve as an important first step towards building models that can accurately perceive and understand the complex dynamics of touch.

   
