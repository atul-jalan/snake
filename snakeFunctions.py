import random
import pygame
import numpy as np

def userChoice():
	return int(input("1 to play snake\n2 to test the neural net\n3 to get neural net performance data after training\n4 to test the algorithm used to train the neural net\n5 to train snake using neural net\n6 to train snake using genetic algorithm\n7 to quit\n"))

def snakeStartingPosition(NUM_CELLS):
	x = random.randint(0,NUM_CELLS-1)
	y = random.randint(2,NUM_CELLS-1)
	snake = [[x, y], [x, y-1], [x, y-2]] #The starting position of the snake.

	return snake

def updateSnake(snake, addX, addY):
	#Traverses through the snake list backwards and sets each object (not including the head of the snake) to the one before it.
	for elements in range(len(snake)-1, 0, -1): 
		snake[elements][0] = snake[elements-1][0]
		snake[elements][1] = snake[elements-1][1]
	#The elements of the head of the snake are changed in accordance with the addX and addY variables, which are + or - 1 depending on the key pressed.
	snake[0][0] += addX
	snake[0][1] += addY

	return snake

def updateFood(food, snake, NUM_CELLS):
	food[0][0] = random.randint(0, NUM_CELLS - 1)
	food[0][1] = random.randint(0, NUM_CELLS - 1)

	#Confirms that the food is not in a spot already occupied by the snake. If it is, the updateFood procedure is run again
	for elements in snake:
		if food[0][0] == elements[0] and food[0][1] == elements[1]:
			updateFood(food, snake, NUM_CELLS)
			break #Present so that the last two lines of code are not rerun in the function

	return food 

def eatFood(food, snake):
	if snake[0][0] == food[0][0] and snake[0][1] == food[0][1]:
		return True
	else:
		return False

def addToSnake(food, snake):
	foodCopy = [0, 0] #Have to make a foodCopy so that the last element of snake does not point to food
	for element in range(0, len(food[0])):
		foodCopy[element] = food[0][element]
	snake.append(foodCopy)

	return snake

def loseGame(snake, food, NUM_CELLS):
	#Returns true if game is lost. False if game is not lost.
	returnBoolean = False

	#Checks if the snake has run into a wall
	if snake[0][0] == -1 or snake[0][0] == NUM_CELLS or snake[0][1] == -1 or snake[0][1] == NUM_CELLS:
		return True

	#Checks if the snake has run into itself by iterating through all elements of the snake starting with the second and checking each against the head of the snake.
	for element in range(1, len(snake)):
		if snake[0][0] == snake[element][0] and snake[0][1] == snake[element][1]:
			return True

	return returnBoolean

def drawObject(object, color, window, cellSize):
	for element in object:
		pygame.draw.rect(window, color, (element[0] * cellSize, element[1] * cellSize, cellSize, cellSize))
	pygame.display.update()

def writeDataToFile(filename, data):
	import csv
	with open(filename,"w") as my_csv:
		csvWriter = csv.writer(my_csv, delimiter = ',')
		csvWriter.writerows(data)
	my_csv.close()
	print (filename, "is written")

def getData(food, snake, NUM_CELLS):
	#There are 12 points in the data list. There are 3 data points for the 4 directions. 3 * 4 = 12.
	#The order in which the list is organized is up, right, down, left. The 3 data points for each
	#direction (in order) are  distance to food, distance to the wall, and distance from another part
	#of itself. The distance being from the head of the snake to each of those 3 points. Each one of
	#the data points is normalized from 0 to 1. A value of 0 means that there is no data for that point.
	#For example, if there is no food above the snake, that data point would be 0. As distance is reduced,
	#the data point gets closer to a value of 1 (formula is in normalizeDistance function)

	data = np.zeros(12)
	#Max distance value here is NUM_CELLS because max distance between snake and food is NUM_CELLS - 1
	#The value is always one greater than max distance because if the max distance was achieved,
	#we want the normalized distance to be slightly greather than 0. A normalized value of 0 
	#should only occur when there is no food above snake or other snake above the snake.

	# Use if you want the snake to always know where the food is.
	if food[0][1] < snake[0][1]: #Food above snake
		data[0] = normalizeDistance(food[0][1], snake[0][1], NUM_CELLS)
	elif food[0][1] > snake[0][1]: #Food below snake
		data[6] = normalizeDistance(food[0][1], snake[0][1], NUM_CELLS)
	if food[0][0] > snake[0][0]: #Food right of snake
		data[3] = normalizeDistance(food[0][0], snake[0][0], NUM_CELLS)
	elif food[0][0] < snake[0][0]: #Food left of snake
		data[9] = normalizeDistance(food[0][0], snake[0][0], NUM_CELLS)

	#Use if you want the snake to only where the food is when that food is directly in (front, left, right, back) of snake.
	# if food[0][0] == snake[0][0] and food[0][1] < snake[0][1]: #Food above snake
	# 	data[0] = normalizeDistance(food[0][1], snake[0][1], NUM_CELLS)
	# elif food[0][0] == snake[0][0] and food[0][1] > snake[0][1]: #Food below snake
	# 	data[6] = normalizeDistance(food[0][1], snake[0][1], NUM_CELLS)
	# elif food[0][1] == snake[0][1] and food[0][0] > snake[0][0]: #Food right of snake
	# 	data[3] = normalizeDistance(food[0][0], snake[0][0], NUM_CELLS)
	# elif food[0][1] == snake[0][1] and food[0][0] < snake[0][0]: #Food left of snake
	# 	data[9] = normalizeDistance(food[0][0], snake[0][0], NUM_CELLS)

	#Max distance value here is NUM_CELLS + 1 because the max distance between snake and wall is NUM_CELLS
	data[1] = normalizeDistance(-1, snake[0][1], NUM_CELLS + 1) #Upper wall and snake
	data[7] = normalizeDistance(NUM_CELLS, snake[0][1], NUM_CELLS + 1) #Lower wall and snake
	data[4] = normalizeDistance(NUM_CELLS, snake[0][0], NUM_CELLS + 1) #Right wall and snake
	data[10] = normalizeDistance(-1, snake[0][0], NUM_CELLS + 1) #Left wall and snake

	#Max distance value here is NUM_CELLS because max distance between snake and other part of snake is NUM_CELLS - 1
	for counter in range(1, len(snake)):
		x = normalizeDistance(snake[counter][0], snake[0][0], NUM_CELLS)
		y = normalizeDistance(snake[counter][1], snake[0][1], NUM_CELLS)

		if snake[counter][0] == snake[0][0] and snake[counter][1] < snake[0][1]: #Snake point above head of snake
			if y > data[2]:
				data[2] = y
		elif snake[counter][0] == snake[0][0] and snake[counter][1] > snake[0][1]: #Snake point below head of snake
			if y > data[8]:
				data[8] = y
		elif snake[counter][1] == snake[0][1] and snake[counter][0] > snake[0][0]: #Snake point right of head of snake
			if x > data[5]:
				data[5] = x
		elif snake[counter][1] == snake[0][1] and snake[counter][0] < snake[0][0]: #Snake point left of head of snake
			if x > data[11]:
				data[11] = x

	return data 

def normalizeDistance(x, y, maxDistance):
	z = x - y
	z = abs(z)
	z = z / maxDistance
	z = 1 - z

	return z

def getScore(snake):
	return len(snake)

def checkIfEqual(List1, List2): #Checks if two 1 dimensional lists are the same
	equal = True
	for x in range(0, len(List1)):
		if List1[x] != List2[x]:
			equal = False
			break

	return equal

def getMoveChoice(food, snake, NUM_CELLS, retries=0):
	#return 0 for up, 1 for right, 2 for down, 3 for left

	# #Simple algo - moves randomly unless doing so results in death, if so, then repicks another random number
	# x = random.randint(0, 3)
	# if checkLoss(food, snake, NUM_CELLS, x) == True:
	# 	retries += 1 #if this loop runs move than 50 times, its likely that no move is possible and thus the loop should be terminated
	# 	if retries < 50:
	# 		x = getMoveChoice(food, snake, NUM_CELLS, retries)
	# return x

	#Advanced Algo - always goes in direction of food, except for when snake or wall is in the way of the food
	if snake[1][0] == snake[0][0] and snake[1][1] < snake[0][1]: #Snake point above head of snake
		if food[0][1] > snake[0][1] and food[0][0] == snake[0][0]: 
			if checkLoss(food, snake, NUM_CELLS, 2) == False:
				return 2
			elif checkLoss(food, snake, NUM_CELLS, 1) == False:
				return 1
			else:
				return 3
		elif food[0][0] > snake[0][0]:
			if checkLoss(food, snake, NUM_CELLS, 1) == False:
				return 1
			elif checkLoss(food, snake, NUM_CELLS, 2) == False:
				return 2
			else:
				return 3
		elif food[0][0] < snake[0][0]:
			if checkLoss(food, snake, NUM_CELLS, 3) == False:
				return 3
			elif checkLoss(food, snake, NUM_CELLS, 2) == False:
				return 2
			else:
				return 1
		else:
			if checkLoss(food, snake, NUM_CELLS, 1) == True and checkLoss(food, snake, NUM_CELLS, 3) == True:
				return 2
			elif checkLoss(food, snake, NUM_CELLS, 1) == False:
				return 1
			else:
				return 3
	elif snake[1][0] == snake[0][0] and snake[1][1] > snake[0][1]: #Snake point below head of snake
		if food[0][1] < snake[0][1] and food[0][0] == snake[0][0]:
			if checkLoss(food, snake, NUM_CELLS, 0) == False:
				return 0
			elif checkLoss(food, snake, NUM_CELLS, 1) == False:
				return 1
			else:
				return 3
		elif food[0][0] > snake[0][0]:
			if checkLoss(food, snake, NUM_CELLS, 1) == False:
				return 1
			elif checkLoss(food, snake, NUM_CELLS, 0) == False:
				return 0
			else:
				return 3
		elif food[0][0] < snake[0][0]:
			if checkLoss(food, snake, NUM_CELLS, 3) == False:
				return 3
			elif checkLoss(food, snake, NUM_CELLS, 0) == False:
				return 0
			else:
				return 1
		else:
			if checkLoss(food, snake, NUM_CELLS, 1) == True and checkLoss(food, snake, NUM_CELLS, 3) == True:
				return 0
			elif checkLoss(food, snake, NUM_CELLS, 1) == False:
				return 1
			else:
				return 3
	elif snake[1][1] == snake[0][1] and snake[1][0] > snake[0][0]: #Snake point right of head of snake
		if food[0][0] < snake[0][0] and food[0][1] == snake[0][1]:
			if checkLoss(food, snake, NUM_CELLS, 3) == False:
				return 3
			elif checkLoss(food, snake, NUM_CELLS, 0) == False:
				return 0
			else:
				return 2
		elif food[0][1] > snake[0][1]:
			if checkLoss(food, snake, NUM_CELLS, 2) == False:
				return 2
			elif checkLoss(food, snake, NUM_CELLS, 3) == False:
				return 3
			else:
				return 0
		elif food[0][1] < snake[0][1]:
			if checkLoss(food, snake, NUM_CELLS, 0) == False:
				return 0
			elif checkLoss(food, snake, NUM_CELLS, 3) == False:
				return 3
			else:
				return 2
		else:
			if checkLoss(food, snake, NUM_CELLS, 2) == True and checkLoss(food, snake, NUM_CELLS, 0) == True:
				return 3
			elif checkLoss(food, snake, NUM_CELLS, 2) == False:
				return 2
			else:
				return 0
	elif snake[1][1] == snake[0][1] and snake[1][0] < snake[0][0]: #Snake point left of head of snake
		if food[0][0] > snake[0][0] and food[0][1] == snake[0][1]:
			if checkLoss(food, snake, NUM_CELLS, 1) == False:
				return 1
			elif checkLoss(food, snake, NUM_CELLS, 0) == False:
				return 0
			else:
				return 2
		elif food[0][1] > snake[0][1]:
			if checkLoss(food, snake, NUM_CELLS, 2) == False:
				return 2
			elif checkLoss(food, snake, NUM_CELLS, 1) == False:
				return 1
			else:
				return 0
		elif food[0][1] < snake[0][1]:
			if checkLoss(food, snake, NUM_CELLS, 0) == False:
				return 0
			elif checkLoss(food, snake, NUM_CELLS, 1) == False:
				return 1
			else:
				return 2
		else:
			if checkLoss(food, snake, NUM_CELLS, 2) == True and checkLoss(food, snake, NUM_CELLS, 0) == True:
				return 1
			elif checkLoss(food, snake, NUM_CELLS, 2) == False:
				return 2
			else:
				return 0

def checkLoss(food, snake, NUM_CELLS, moveChoice):
	if moveChoice == 0:
		if snake[0][1] == 0:
			return True

		for element in range(1, len(snake)):
			if snake[0][0] == snake[element][0] and snake[0][1] - 1 == snake[element][1]:
				return True

	elif moveChoice == 1:
		if snake[0][0] == NUM_CELLS -1:
			return True

		for element in range(1, len(snake)):
			if snake[0][0] + 1 == snake[element][0] and snake[0][1] == snake[element][1]:
				return True

	elif moveChoice == 2:
		if snake[0][1] == NUM_CELLS - 1:
			return True

		for element in range(1, len(snake)):
			if snake[0][0] == snake[element][0] and snake[0][1] + 1 == snake[element][1]:
				return True

	elif moveChoice == 3:
		if snake[0][0] == 0:
			return True

		for element in range(1, len(snake)):
			if snake[0][0] - 1 == snake[element][0] and snake[0][1] == snake[element][1]:
				return True

	return False
