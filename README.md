Uses raytracing to simmulate the view of a camera looking at a black hole. The skybox (background of the black hole) is <img width="442" height="442" alt="skybox" src="https://github.com/user-attachments/assets/9651fd35-05a0-422e-9139-898da86bccfc" />


(Stored in `skybox.png`).

The resulting image (in 1000x1000 px resolution) is <img width="1000" height="1000" alt="blackhole" src="https://github.com/user-attachments/assets/8b628def-0dc8-4727-89fa-7ee2d6357e45" />

  Dependencies: `numpy`, `scipy`, `Pillow` and `tqdm`

  Notice the variables `y_px` and `x_px` control the resolution of the output image, and correspondingly can increase runtime drastically. By default, these variables are set to 1000, which will take quite a while to complete.
