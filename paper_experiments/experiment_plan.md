* General rules  
  * For each job, make a folder inside paper\_experiments  
  * Also, write an html file that summarizes each job’s experiment, along with an md file that will keep experiment result information useful for writing the paper  
    * Examples of useful information will be metrics and tables that could be directly ported to the latex files in paper\_source  
  * Don’t write anything into paper\_source/. Store the results in paper\_experiments  
  * For qualitative results, store the assets and results inside log/ with a recognizable folder name for each job  
  * Our method will use the following  
    * Retrieval: DINOv3-based retrieval summarized in transfer\_pipeline.py  
    * Coarse alignment using superpoint \+ superglue with normals at scale 4x as the default (refer to train\_refine\_scripts/transfer\_all\_real\_data\_gt\_retrieval\_tactile\_normal/)  
    * Neural network-based refinement using RebotNet \+ temporal FiLM \+ normal map concatenation  
      * Pre-trained network is in log/rebot\_checkpoints\_S\_geomcat\_film  
* Local Jobs  
  * Summarize statistics for dataset generation  
  * Ground-truth retrieval two-column figure creation  
    * Save all the image assets used for making the figure  
  * Application in 3D Surface Reconstruction and Visuo-tactile Sensor Simulation  
    * Make figure of 3D surface reconstruction and visuo-tactile sensor simulation  
      * A two-column figure that shows the 3D reconstruction results of our method  
        * Take an example from the ground-truth retrieval benchmark data  
      * The first row will show the reference tactile normal video  
      * The second row will show predicted tactile normal video after neural network inference  
      * The third row will show the heightmap 3D point clouds per frame   
      * The fourth row will show the RGB visuo-tactile video frames  
      * Each column will be the video frames  
* Dirac Jobs  
  * Ground-truth retrieval performance comparison of our method against the baselines  
    * Include metrics both for coarse transfer and refined transfer using neural network   
    * Run only on 50 eval objects  
    * Exclude TaRF results for now since a new model is being trained  
      * I will ask for running this once we have trained TaRF models available  
    * Ground-truth retrieval qualitative result comparison  
      * Extract assets for each baseline  
      * We want a two-column figure that shows the prediction results of our method on various touch locations  
        * The figure itself will be a matrix of shape 3 x 6  
          * 1st column: reference touch mid frame  
          * 2nd column: reference normal mid frame  
          * 3rd column: query normal mid frame  
          * 4th column: query touch coarse transfer  
          * 5th column: query touch refined transfer using network  
          * 6th column: ground-truth query touch normal  
  * Full pipeline benchmark performance comparison of our method against the baselines  
    * Select 3\~5 query touch per object and set the rest as reference  
    * Exclude TaRF results for now since a new model is being trained  
      * I will ask for running this once we have trained TaRF models available  
    * Qualitative result comparison  
      * Extract assets for each baseline  
      * The figure will be a matrix  
        * Each row will be one method including ours  
          * The first row will show the reference tactile normal video  
        * Each column will be video frames predicted from the tested methods  
          * For image-only predictions, we will put the image on the first frame and write N/A (image only) on the rest of the frames  
  * Ablation Study  
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
        * Use the coarse alignment results as in transfer\_all\_real\_data\_gt\_retrieval\_tactile\_normal/  
      * w/o Temporal FiLM  
        * Use the pre-trained network log/rebot\_checkpoints\_S\_geomcat\_none  
      * w/o Normal concatenation in network-based refinement  
        * Use the pre-trained network log/rebot\_checkpoints\_S\_pseudo\_mini\_tactile\_normal\_superpoint\_superglue\_cond-film-normal  
      * You won’t need to train any new networks here  
    * Make a small one-column figure that compares the neural network predictions from after various ablations.   
      * w/o Neural network-based refinement  
      * w/o Temporal FiLM  
      * w/o Normal concatenation in network-based refinement  
  * Runtime analysis  
    * Use an object from the full pipeline benchmark data  
    * Given N reference touches and 1 query, report the time it takes to produce a touch prediction in the following format  
      * Retrieval phase after DINOv3 feature extraction takes Xs  
      * Coarse alignment after local feature matching takes Xs  
      * Neural network-based refinement takes Xs per frame

