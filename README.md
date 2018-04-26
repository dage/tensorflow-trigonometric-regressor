# tensorflow-trigonometric-regressor
Uses tensorflow DNN regressor estimator to model a trigonometric equation.

This is a simple example of using the tensorflow high-level estimator API to model a 1-dimensional trigonometric equation. It tries different DNN structures to look both at performance aspects and how well the different structures models the equation. Each DNN structure is stored as a different sub-directory in the model directory, so use tensorboard with the following command in a shell to examine the results:  `tensorboard --logdir="model_dir"`
