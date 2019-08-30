import numpy as np
import random

def getBestModels(NUM_PARENTS, modelsFitness, Models, genNumber):
	import heapq
	bestScores = heapq.nlargest(NUM_PARENTS, range(len(modelsFitness)), modelsFitness.take)
	bestModels = []
	print ("GENERATION", genNumber)
	for x in range(0, NUM_PARENTS):
		print(modelsFitness[bestScores[x]], "index:", bestScores[x])
		bestModels.append(Models[bestScores[x]])

	return bestModels
	
def getNextGeneration(CHILDS_PER_PARENT, NUM_PARENTS, models, MUTATION_RATE, type, genNumber, NUM_ELITE=0):
	newModels = []
	for x in range(0, NUM_PARENTS):
		if type == "mutation":
			for y in range(0, CHILDS_PER_PARENT):
				newModels.append(mutate(models[x], MUTATION_RATE))

		elif type == "midpoint_crossover":
			breedingPartners = random.sample(range(0, NUM_PARENTS), CHILDS_PER_PARENT)
			for partner in breedingPartners:
				newModels.append(midpointCrossover(models[x], models[partner], MUTATION_RATE))

		elif type == "uniform_crossover":
			breedingPartners = random.sample(range(0, NUM_PARENTS), CHILDS_PER_PARENT)
			for partner in breedingPartners:
				newModels.append(uniformCrossover(models[x], models[partner], MUTATION_RATE))

		elif type == "midpoint_crossover":
			breedingPartners = random.sample(range(0, NUM_PARENTS), CHILDS_PER_PARENT)
			for partner in breedingPartners:
				newModels.append(midpointCrossover(models[x], models[partner], MUTATION_RATE))

	for y in range(0, NUM_ELITE):
		newModels.pop()

	for y in range (0, NUM_ELITE):
		newModels.append(models[y])

	print ("New models generated for generation", genNumber)
	return newModels

def mutate(model, MUTATION_RATE):
	weightA = np.array(model.get_weights())
	randNumber = random.randint(1, 1/MUTATION_RATE)
	listA1 = weightA[0]
	listA2 = weightA[2]
	listA3 = weightA[4]

	for x in range(0, 12):
		for y in range(0, 16):
			if(random.randint(1, 1/MUTATION_RATE) == randNumber):
				listA1[x][y] = listA1[x][y] * np.random.uniform(0.5, 1.5)

	for x in range(0, 16):
		for y in range(0, 16):
			if(random.randint(1, 1/MUTATION_RATE) == randNumber):
				listA2[x][y] = listA2[x][y] * np.random.uniform(0.5, 1.5)

	for x in range(0, 16):
		for y in range(0, 4):
			if(random.randint(1, 1/MUTATION_RATE) == randNumber):
				listA3[x][y] = listA3[x][y] * np.random.uniform(0.5, 1.5)

	model.layers[0].set_weights([listA1, weightA[1]])
	model.layers[1].set_weights([listA2, weightA[3]])
	model.layers[2].set_weights([listA3, weightA[5]])
	
	# print (weightA)
	# print ("\n\n\n\n\n\n")
	# print (model.get_weights())
	# print ("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
	return model

def averageCrossover(modelA, modelB, MUTATION_RATE):
	weightA = np.array(modelA.get_weights())
	weightB = modelB.get_weights()

	listA1 = weightA[0]
	listA2 = weightA[2]
	listA3 = weightA[4]
	listB1 = weightB[0]
	listB2 = weightB[2]
	listB3 = weightB[4]

	newList1 = np.zeros(192).reshape(12, 16)
	newList2 = np.zeros(256).reshape(16, 16)
	newList3 = np.zeros(64).reshape(16, 4)

	for x in range(0, 12):
		for y in range(0, 16):
			newList1[x][y] = (listA1[x][y] + listB1[x][y]) / 2

	for x in range(0, 16):
		for y in range(0, 16):
			newList1[x][y] = (listA2[x][y] + listB2[x][y]) / 2

	for x in range(0, 16):
		for y in range(0, 4):
			newList1[x][y] = (listA3[x][y] + listB3[x][y]) / 2

	modelA.layers[0].set_weights([newList1, weightA[1]])
	modelA.layers[1].set_weights([newList2, weightA[3]])
	modelA.layers[2].set_weights([newList3, weightA[5]])
	
	return modelA

def midpointCrossover(modelA, modelB, MUTATION_RATE):
	weightA = np.array(modelA.get_weights())
	weightB = modelB.get_weights()

	randNumber = random.randint(1, 1/MUTATION_RATE)

	listA1 = weightA[0]
	listA2 = weightA[2]
	listA3 = weightA[4]
	listB1 = weightB[0]
	listB2 = weightB[2]
	listB3 = weightB[4]

	newList1 = np.zeros(192).reshape(12, 16)
	newList2 = np.zeros(256).reshape(16, 16)
	newList3 = np.zeros(64).reshape(16, 4)

	for x in range(0, 12):
		for y in range(0, 16):
			if(x < 6):
				if(random.randint(1, 1/MUTATION_RATE) == randNumber):
					newList1[x][y] = np.random.uniform(-1.0, 1.0)
				else:
					newList1[x][y] = listA1[x][y]
			else:
				if(random.randint(1, 1/MUTATION_RATE) == randNumber):
					newList1[x][y] = np.random.uniform(-1.0, 1.0)
				else:
					newList1[x][y] = listB1[x][y]

	for x in range(0, 16):
		for y in range(0, 16):

			if(x < 8):
				if(random.randint(1, 1/MUTATION_RATE) == randNumber):
					newList2[x][y] = np.random.uniform(-1.0, 1.0)
				else:
					newList2[x][y] = listA2[x][y]
			else:
				if(random.randint(1, 1/MUTATION_RATE) == randNumber):
					newList2[x][y] = np.random.uniform(-1.0, 1.0)
				else:
					newList2[x][y] = listB2[x][y]

	for x in range(0, 16):
		for y in range(0, 4):
			if(x < 8):
				if(random.randint(1, 1/MUTATION_RATE) == randNumber):
					newList3[x][y] = np.random.uniform(-1.0, 1.0)
				else:
					newList3[x][y] = listA3[x][y]
			else:
				if(random.randint(1, 1/MUTATION_RATE) == randNumber):
					newList3[x][y] = np.random.uniform(-1.0, 1.0)
				else:
					newList3[x][y] = listB3[x][y]

	modelA.layers[0].set_weights([newList1, weightA[1]])
	modelA.layers[1].set_weights([newList2, weightA[3]])
	modelA.layers[2].set_weights([newList3, weightA[5]])
	
	return modelA

def uniformCrossover(modelA, modelB, MUTATION_RATE):
	weightA = np.array(modelA.get_weights())
	weightB = modelB.get_weights()

	randNumber = random.randint(1, 1/MUTATION_RATE)

	listA1 = weightA[0]
	listA2 = weightA[2]
	listA3 = weightA[4]
	listB1 = weightB[0]
	listB2 = weightB[2]
	listB3 = weightB[4]

	newList1 = np.zeros(192).reshape(12, 16)
	newList2 = np.zeros(256).reshape(16, 16)
	newList3 = np.zeros(64).reshape(16, 4)

	for x in range(0, 12):
		for y in range(0, 16, 2):
			if(random.randint(1, 1/MUTATION_RATE) == randNumber):
				newList1[x][y] = np.random.uniform(-1.0, 1.0)
			else:
				newList1[x][y] = listA1[x][y]

			if(random.randint(1, 1/MUTATION_RATE) == randNumber):
				newList1[x][y+1] = np.random.uniform(-1.0, 1.0)
			else:
				newList1[x][y+1] = listB1[x][y+1]

	for x in range(0, 16):
		for y in range(0, 16, 2):
			if(random.randint(1, 1/MUTATION_RATE) == randNumber):
				newList2[x][y] = np.random.uniform(-1.0, 1.0)
			else:
				newList2[x][y] = listA2[x][y]
				
			if(random.randint(1, 1/MUTATION_RATE) == randNumber):
				newList2[x][y+1] = np.random.uniform(-1.0, 1.0)
			else:
				newList2[x][y+1] = listB2[x][y+1]

	for x in range(0, 16):
		for y in range(0, 4, 2):
			if(random.randint(1, 1/MUTATION_RATE) == randNumber):
				newList3[x][y] = np.random.uniform(-1.0, 1.0)
			else:
				newList3[x][y] = listA3[x][y]
				
			if(random.randint(1, 1/MUTATION_RATE) == randNumber):
				newList3[x][y+1] = np.random.uniform(-1.0, 1.0)
			else:
				newList3[x][y+1] = listB3[x][y+1]

	modelA.layers[0].set_weights([newList1, weightA[1]])
	modelA.layers[1].set_weights([newList2, weightA[3]])
	modelA.layers[2].set_weights([newList3, weightA[5]])
	
	return modelA

def getFitnessScore(individualGame):

	fitnessScore = (individualGame[2] - 3) * 100

	#The scores increase as the snake gets closer to food
	postMoveFoodScore = individualGame[0][0][0] + individualGame[0][0][3] + individualGame[0][0][6] + individualGame[0][0][9]

	for move in range(0, len(individualGame[0])-1):
		prevMoveFoodScore = postMoveFoodScore
		postMoveFoodScore = individualGame[0][move+1][0] + individualGame[0][move+1][3] + individualGame[0][move+1][6] + individualGame[0][move+1][9]
		if (postMoveFoodScore > prevMoveFoodScore):
			fitnessScore += 1.5
		elif (postMoveFoodScore < prevMoveFoodScore):
			fitnessScore += -1

	return fitnessScore