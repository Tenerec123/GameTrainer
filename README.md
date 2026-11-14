# GameTrainer
A library to create and train an AI to play games such us Chess, Checkers, Go or Strategeti. It implements the MCTS, the model creation and the trainloop.

The only work it requires is to implement the game engine, which includes the position class, the conversion of that position into parameters, the conversion of model output into moves and the decider of which moves are legal and which ones are not.

install it with: pip install git+https://github.com/tenerec123/GameTrainer.git