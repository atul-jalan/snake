def visualSnake(type, speed=10):
	import pygame
	pygame.init()
	clock = pygame.time.Clock() 	

	red = (255, 0, 0)
	white = (255, 255, 255)
	black = (0, 0, 0)
	windowSize = 500
	cellSize = windowSize / NUM_CELLS

	window = pygame.display.set_mode((windowSize, windowSize))
	
	snake = sf.snakeStartingPosition(NUM_CELLS)
	food = [[0, 0]]
	xDirection = 0
	yDirection = 1

	window.fill(white)
	food = sf.updateFood(food, snake, NUM_CELLS)
	sf.drawObject(snake, red, window, cellSize)
	sf.drawObject(food, black, window, cellSize)

	if type == "test_net":
		model = load_model('initialModel.h5')
		reshaped = np.reshape(sf.getData(food, snake, NUM_CELLS), (1, 12))
		prediction = model.predict(reshaped)
		prediction = prediction.flatten()
		choice = np.argmax(prediction)
	elif type == "test_algo":
		choice = sf.getMoveChoice(food, snake, NUM_CELLS)

	endGame = False
	while endGame == False:

		if type == "play":
			keyPressed = False
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					endGame = True
				if event.type == pygame.KEYDOWN:
						keyPressed = True
						window.fill(white)
						sf.drawObject(food, black, window, cellSize)

						if event.key == pygame.K_RIGHT:
							snake = sf.updateSnake(snake, 1, 0)
							xDirection = 1
							yDirection = 0
						elif event.key == pygame.K_LEFT:
							snake = sf.updateSnake(snake, -1, 0)
							xDirection = -1
							yDirection = 0
						elif event.key == pygame.K_UP:
							snake = sf.updateSnake(snake, 0, -1)
							xDirection = 0
							yDirection = -1
						elif event.key == pygame.K_DOWN:
							snake = sf.updateSnake(snake, 0, 1)
							xDirection = 0
							yDirection = 1

			if keyPressed == False:
				window.fill(white)
				sf.drawObject(food, black, window, cellSize)
				snake = sf.updateSnake(snake, xDirection, yDirection)

			if sf.loseGame(snake, food, NUM_CELLS) == True:
				endGame = True
			if sf.eatFood(food, snake) == True:
				snake = sf.addToSnake(food, snake)
				food = sf.updateFood(food, snake, NUM_CELLS)
			sf.drawObject(snake, red, window, cellSize)

			clock.tick(speed)

		elif type == "test_net" or type == "test_algo":
			window.fill(white)
			sf.drawObject(food, black, window, cellSize)

			if choice == 0:
				snake = sf.updateSnake(snake, 0, -1)
			elif choice == 1:
				snake = sf.updateSnake(snake, 1, 0)
			elif choice == 2:
				snake = sf.updateSnake(snake, 0, 1)
			elif choice == 3:
				snake = sf.updateSnake(snake, -1, 0)

			if sf.loseGame(snake, food, NUM_CELLS) == True:
				endGame = True
			if sf.eatFood(food, snake) == True:
				snake = sf.addToSnake(food, snake)
				food = sf.updateFood(food, snake, NUM_CELLS)
			sf.drawObject(snake, red, window, cellSize)

			if type == "test_net":
				reshaped = np.reshape(sf.getData(food, snake, NUM_CELLS), (1, 12))
				prediction = model.predict(reshaped)
				prediction = prediction.flatten()
				choice = np.argmax(prediction)	
			elif type == "test_algo":
				choice = sf.getMoveChoice(food, snake, NUM_CELLS)

			pygame.time.delay(10)	

	print (sf.getScore(snake))
	pygame.quit()

def nonVisualSnake(type, model = [], dataSetSize=500000):
	progressBar = dataSetSize / 100
	progress = 1
	numGamesPlayed = 0
	totalScore = 0
	returnData = [[], [], 0]
	allData = [[], []]

	while len(allData[0]) < dataSetSize:
		if len(allData[0]) > progressBar:
			print (progress, "percent data compiled")
			progress = int((len(allData[0])/dataSetSize) * 100)
			progressBar += dataSetSize / 100

		returnData = nonVisualSnakeIndividualGame([], [], type, model)
		numGamesPlayed += 1
		totalScore += returnData[2]
		returnData[0].pop() #Removes the last data point because that data point is the move the snake made
		returnData[1].pop() #that made it lose. That data is unwanted to train the snake.

		for data in range(0, len(returnData[0])):
			allData[0].append(returnData[0][data])
			allData[1].append(returnData[1][data])

	avgScore = totalScore / numGamesPlayed
	print ("All data compiled.", numGamesPlayed, "games played with average score", avgScore)
	print (len(allData[0]), "   ", len(allData[1]))
	sf.writeDataToFile("MovementData.csv", allData[0])
	sf.writeDataToFile("ChoiceData.csv", allData[1])

def nonVisualSnakeIndividualGame(movementData, correctChoice, type, model = []):

	snake = sf.snakeStartingPosition(NUM_CELLS)
	food = sf.updateFood([[0, 0]], snake, NUM_CELLS)

	rpttnData = [] #repetition data - used and explained below
	lklyRepeated = [] #likely repeated - used and explained below

	endGame = False
	while endGame == False:

		movementData.append(sf.getData(food, snake, NUM_CELLS))

		if type == "use_net":
			reshaped = np.reshape(sf.getData(food, snake, NUM_CELLS), (1, 12))
			prediction = model.predict(reshaped)
			prediction = prediction.flatten()
			choice = np.argmax(prediction)
		elif type == "use_algo":
			choice = sf.getMoveChoice(food, snake, NUM_CELLS)

		moveChoiceList = [0, 0, 0, 0]
		if choice == 0:
			snake = sf.updateSnake(snake, 0, -1)
			moveChoiceList[0] = 1
		elif choice == 1:
			snake = sf.updateSnake(snake, 1, 0)
			moveChoiceList[1] = 1
		elif choice == 2:
			snake = sf.updateSnake(snake, 0, 1)
			moveChoiceList[2] = 1
		elif choice == 3:
			snake = sf.updateSnake(snake, -1, 0)
			moveChoiceList[3] = 1

		correctChoice.append(moveChoiceList)

		if sf.loseGame(snake, food, NUM_CELLS) == True:
			endGame = True
		if sf.eatFood(food, snake) == True:
			snake = sf.addToSnake(food, snake)
			food = sf.updateFood(food, snake, NUM_CELLS)

		#If two elements are the same, they're added to the lklyrepeated array, with the distance between them as a data point in that array
		#if that element shows up again and the distance between that element and the last time it occurs is the same as the one stored
		#in lklyrepeated, then the game is exited due to an infinite loop
		if len(movementData) % 144 == 0: #Checks for repetition every 144 moves
			rpttnData.append(movementData[-1]) #logs every 144th move into list rppttnData
			for x in range(len(rpttnData)-2 , -1, -1): #iterates through every element backwards except the newest element in the array
				if sf.checkIfEqual(rpttnData[x], rpttnData[-1]) == True: #because the newest element is compared against each element before it
					for y in range(0, len(lklyRepeated)):
						if sf.checkIfEqual(rpttnData[-1], lklyRepeated[y][0]) and len(rpttnData)-1-x == lklyRepeated[y][1]:
							print ("Exited game due to infinite loop")
							endGame = True

					lklyRepeated.append([rpttnData[x], len(rpttnData)-1-x])

	return [movementData, correctChoice, sf.getScore(snake)]

def trainModel(type):

	model = Sequential()
	model.add(Dense(16, input_dim = 12, activation = 'relu'))
	model.add(Dense(16, activation = 'relu'))
	model.add(Dense(4, activation = 'sigmoid'))
	model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

	if type == "train_net":
		dataX = loadtxt('MovementData.csv', delimiter = ',')
		dataY = loadtxt('ChoiceData.csv', delimiter = ',')
		model.fit(dataX, dataY, epochs = 10, batch_size = 100)
		model.save('initialModel.h5')
	elif type == "generate_net":
		return model

def geneticAlgorithm():
	POPULATION_SIZE = 80
	PERCENT_BEST = .25
	NUM_PARENTS = int(POPULATION_SIZE * PERCENT_BEST)
	CHILDS_PER_PARENT = int(POPULATION_SIZE/NUM_PARENTS)
	NUM_ELITE = 4
	MUTATION_RATE = .1
	NUM_GENERATIONS = 100
	numGamesPerModel = 10	

	print ("Beginning generation 0")
	models = getBaseModels(POPULATION_SIZE)

	for x in range(0, NUM_GENERATIONS):
		modelsFitness = getModelFitnessScores(POPULATION_SIZE, numGamesPerModel, models, x)
		bestModels = gaf.getBestModels(NUM_PARENTS, modelsFitness, models, x)

		bestModels[0].save('initialModel.h5')
		print ("Model saved")

		print ("Beginning generation", x+1)
		models = gaf.getNextGeneration(CHILDS_PER_PARENT, NUM_PARENTS, bestModels, MUTATION_RATE, "mutation", x+1, NUM_ELITE)

def getBaseModels(POPULATION_SIZE):
	baseModels = []
	for x in range(0, POPULATION_SIZE):
		print ("Initializing Model", x + 1)
		baseModels.append(trainModel("generate_net"))
	print ("Initial models created")

	return baseModels

def getModelFitnessScores(POPULATION_SIZE, numGamesPerModel, model, genNumber):
	modelFitness = np.zeros(POPULATION_SIZE)

	for x in range(0, POPULATION_SIZE):
		print ("testing model", x)
		modelFitness[x] = indivGameFitnessScore(model[x], numGamesPerModel)

	print ("models tested for generation", genNumber)
	return modelFitness

def indivGameFitnessScore(model, numGamesPerModel):
	fitnessScore = 0
	for y in range(0, numGamesPerModel):
		individualGame = nonVisualSnakeIndividualGame([], [], "use_net", model)
		fitnessScore += gaf.getFitnessScore(individualGame)

	return fitnessScore



import snakeFunctions as sf
import geneticAlgoFunctions as gaf
import random
import time

import numpy as np
from numpy import loadtxt
from numpy import argsort

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.utils import to_categorical

NUM_CELLS = 20

loop = True
while loop:
	userResponse = sf.userChoice()

	if userResponse == 1: #play snake
		visualSnake("play")
	elif userResponse == 2: #test snake
		visualSnake("test_net")
	elif userResponse == 3: #get score data for snake
		nonVisualSnake("use_net", load_model('initialModel.h5'), 100000)
	elif userResponse == 4: #test algo
		visualSnake("test_algo")
	elif userResponse == 5: #train snake
		nonVisualSnake("use_algo")
		trainModel("train_net")
	elif userResponse == 6: #genetic algorithm
		geneticAlgorithm()
	elif userResponse == 7: #quit
		quit()