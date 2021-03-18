# %%writefile lazyGoose.py
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col

import random
import numpy as np
from queue import Queue

class Consts:
    ROWS = 7
    COLS = 11
    DX = [0, 1, 0, -1, 0]
    DY = [1, 0, -1, 0, 0]
    CODE = [Action.EAST.name, Action.SOUTH.name, Action.WEST.name, Action.NORTH.name, 'NULL']
    MAX_COOL_THRESHOLD = 5 # don't care about more than this number of steps
    MAX_LENGTH = 30 # when this is reached, just run
    MAX_BOARD_SIZE = 40 # when this number of cells is occupied, just run
    MAX_CHASE_STEPS = 14
    MAX_SAFE_STEPS = 7
    LENGTH_AGGRESSIVE = 6 # when this is reached, don't take food that is uncertain
    DEBUG = False

class Logger:
    def log(msg, step=None):
        if Consts.DEBUG:
            if step != None: msg += f", choice={Consts.CODE[step]}"
            print(f"[{Global.playerIndex}] {msg}")
        return step

class Global:
    roundCnt = 0
    food = []
    geese = [[] for i in range(4)]
    playerIndex = None
    prevMoves = [None]*4

    def update(observation, configuration):
        if Global.roundCnt > 0:
            nxtGeese = [[row_col(p, configuration.columns) for p in observation.geese[g]] for g in range(4)]
            for g in range(4):
                if len(nxtGeese[g]) == 0: continue
                Global.prevMoves[g] = [d for d in range(4) if Helper.moveTowards(Global.geese[g][0], d) == nxtGeese[g][0]][0]

        Global.roundCnt += 1
        Global.food = [row_col(f, configuration.columns) for f in observation.food]
        Global.geese = [[row_col(p, configuration.columns) for p in observation.geese[g]] for g in range(4)]
        Global.playerIndex = observation.index

        prevMove = Global.prevMoves[Global.playerIndex]
        if prevMove is None: prevMove = 4
        Logger.log(f"Round #{Global.roundCnt}, length={len(Global.geese[Global.playerIndex])}, prevMove={Consts.CODE[prevMove]}")

class Helper:
    def moveTowards(xy, d):
        x, y = xy
        nx = (x + Consts.DX[d] + Consts.ROWS) % Consts.ROWS
        ny = (y + Consts.DY[d] + Consts.COLS) % Consts.COLS
        return (nx, ny)

    def getManhattanDis(st, ed):
        a, b = st
        c, d = ed
        if a > c: a, c = c, a
        if b > d: b, d = d, b
        return min(c-a, Consts.ROWS+a-c) + min(d-b, Consts.COLS+b-d)

class Maps:
    def generateHeatMap():
        heatMap = np.zeros((Consts.ROWS, Consts.COLS))
        for g in range(4):
            if len(Global.geese[g]) == 0: continue
            for i in range(len(Global.geese[g])): heatMap[Global.geese[g][i]] = len(Global.geese[g]) - i
        return heatMap

    def generateDisFromPlayers(heatMap, bold):
        dis = np.ones((4, Consts.ROWS, Consts.COLS)) * np.inf
        for g in range(4):
            if len(Global.geese[g]) == 0: continue
            choice = 0
            for it in range(2):
                nearTail = Helper.getManhattanDis(Global.geese[g][0], Global.geese[g][-1]) == 1
                que = Queue()
                que.put(Global.geese[g][0])
                dis[(g,) + Global.geese[g][0]] = 0
                while not que.empty():
                    x, y = que.get()
                    for d in range(4):
                        nx, ny = Helper.moveTowards((x, y), d)
                        # if g != Global.playerIndex and nearTail and Global.geese[g][0] == (x, y):
                        #     if len(Global.geese[g]) == 2:
                        #         if Global.geese[g][-1] == (nx, ny): continue
                        #     if len(Global.geese[g])  > 2:
                        #         if Global.geese[g][-1] != (nx, ny): continue
                        if not bold and it==0: # if it==1 then no choice
                        # if Global.geese[g][0] == (x, y) and sum(len(g)>0 for g in Global.geese) > 3 and any(Global.geese[q][0] == (nx, ny) for q in range(4) if len(Global.geese[q])>len(Global.geese[g])): continue
                            if Global.geese[g][0] == (x, y) and sum(len(g)>0 for g in Global.geese) > 2 and any(Helper.getManhattanDis(Global.geese[q][0], (nx, ny)) == 1 for q in range(4) if g!=q and len(Global.geese[q])>0): continue
                            if Global.geese[g][0] == (x, y) and sum(len(g)>0 for g in Global.geese) < 3 and any(Helper.getManhattanDis(Global.geese[q][0], (nx, ny)) == 1 for q in range(4) if len(Global.geese[q])>len(Global.geese[g])): continue
                        if dis[g, nx, ny] < np.inf: continue
                        if dis[g, x, y]+1 < heatMap[nx, ny]: continue
                        choice += 1
                        dis[g, nx, ny] = dis[g, x, y]+1
                        que.put((nx, ny))
                if choice > 0: break # otherwise have to be bold
        return dis

    def generateCoolMap(playerIndex, disFromPlayers):
        coolMap = np.ones((Consts.ROWS, Consts.COLS)) * np.inf
        for g in range(4):
            if len(Global.geese[g]) == 0: continue
            if g == playerIndex: continue
            for x in range(Consts.ROWS):
                for y in range(Consts.COLS):
                    coolMap[x, y] = min(coolMap[x, y], disFromPlayers[g, x, y])
        return coolMap

    def generateMinHeatDisGuaranteed(playerIndex, heatMap, coolMap):
        minHeatDisGuaranteed = np.zeros(4)
        for td in range(4):
            if (td^2) == Global.prevMoves[playerIndex]: continue
            x, y = Helper.moveTowards(Global.geese[playerIndex][0], td)
            if coolMap[x, y] <= 1 or heatMap[x, y] > 1: continue
            dis = np.ones((Consts.ROWS, Consts.COLS)) * np.inf
            que = Queue()
            minHeatDisGuaranteed[td] = dis[x, y] = 1
            que.put((x, y))
            while not que.empty():
                x, y = que.get()
                for d in range(4):
                    nx, ny = Helper.moveTowards((x, y), d)
                    if dis[x, y]+1 < heatMap[nx, ny]: continue
                    if dis[nx, ny] < np.inf: continue
                    dis[nx, ny] = dis[x, y]+1
                    minHeatDisGuaranteed[td] = dis[x, y]+1
                    que.put((nx, ny))
        return minHeatDisGuaranteed

    def generateMinDisGuaranteed(playerIndex, heatMap, coolMap):
        minDisGuaranteed = np.zeros((Consts.MAX_COOL_THRESHOLD+1, 4))
        for coolThreshold in range(Consts.MAX_COOL_THRESHOLD, -1, -1):
            for td in range(4):
                if (td^2) == Global.prevMoves[playerIndex]: continue
                x, y = Helper.moveTowards(Global.geese[playerIndex][0], td)
                if coolMap[x, y] <= 1 or heatMap[x, y] > 1: continue
                dis = np.ones((Consts.ROWS, Consts.COLS)) * np.inf
                que = Queue()
                minDisGuaranteed[coolThreshold, td] = dis[x, y] = 1
                que.put((x, y))
                while not que.empty():
                    x, y = que.get()
                    for d in range(4):
                        nx, ny = Helper.moveTowards((x, y), d)
                        if dis[x, y]+1 < heatMap[nx, ny]: continue
                        if dis[nx, ny] < np.inf: continue
                        if coolMap[nx, ny] <= coolThreshold and dis[x, y]+1 >= coolMap[nx, ny]: continue
                        dis[nx, ny] = dis[x, y]+1
                        minDisGuaranteed[coolThreshold, td] = dis[x, y]+1
                        que.put((nx, ny))
        return minDisGuaranteed

    def generateHeatMapConsideringFood(disFromPlayers):
        foodDisFromPlayer = [[disFromPlayers[(g,)+f] for f in Global.food] for g in range(4)]
        heatMap = np.zeros((Consts.ROWS, Consts.COLS))
        for g in range(4):
            if len(Global.geese[g]) == 0: continue
            for i in range(len(Global.geese[g])):
                heatMap[Global.geese[g][i]]  = len(Global.geese[g]) - i
                heatMap[Global.geese[g][i]] -= (Global.roundCnt%40 == 0 and i+2 != len(Global.geese[g]))
                for f in range(2):
                    if foodDisFromPlayer[g][f] < min(foodDisFromPlayer[q][f] for q in range(4) if g!=q):
                        heatMap[Global.geese[g][i]] += heatMap[Global.geese[g][i]] >= foodDisFromPlayer[g][f]
        return heatMap

    def generateOppHeatMapConsideringFood(disFromPlayers):
        foodDisFromPlayer = [[disFromPlayers[(g,)+f] for f in Global.food] for g in range(4)]
        heatMap = np.zeros((Consts.ROWS, Consts.COLS))
        for g in range(4):
            if len(Global.geese[g]) == 0: continue
            for i in range(len(Global.geese[g])):
                heatMap[Global.geese[g][i]]  = len(Global.geese[g]) - i
                if g == Global.playerIndex: continue
                heatMap[Global.geese[g][i]] -= (Global.roundCnt%40 == 0 and i+2 != len(Global.geese[g]))
                for f in range(2):
                    if foodDisFromPlayer[g][f] < min(foodDisFromPlayer[q][f] for q in range(4) if g!=q):
                        heatMap[Global.geese[g][i]] += heatMap[Global.geese[g][i]] >= foodDisFromPlayer[g][f]
                # heatMap[Global.geese[g][i]] += heatMap[Global.geese[g][i]] >= foodDisFromPlayerAscending[g][0]
                # heatMap[Global.geese[g][i]] += heatMap[Global.geese[g][i]]-1 >= foodDisFromPlayerAscending[g][1]
        return heatMap

class Decisions:
    def chaseSelf(playerIndex, heatMap, coolMap, mustEat=False): # TODO: consider food prob and roundCnt
        if len(Global.geese[playerIndex]) < 4: return None
        for it in range(2):
            for maxSteps in range(1, Consts.MAX_CHASE_STEPS):
                def explore(playerIndex, xy, occupied, step, foodTaken):
                    if step == maxSteps: return None
                    for d in range(4):
                        nx, ny = Helper.moveTowards(xy, d)
                        if (nx, ny) in occupied: continue # my new steps are colliding
                        if heatMap[nx, ny] + foodTaken * ((nx, ny) in Global.geese[playerIndex]) > step+1: continue
                        shrinks = Global.roundCnt % 40 > (Global.roundCnt + step) % 40
                        if it==1 or foodTaken:
                            if step+1 == heatMap[nx, ny]+foodTaken-shrinks and (nx, ny) in Global.geese[playerIndex]:
                                return [d]
                        if coolMap[nx, ny] <= step+1: continue
                        foodTaken += (nx, ny) in Global.food
                        occupied.append((nx, ny))
                        seq = explore(playerIndex, (nx, ny), occupied, step+1, foodTaken)
                        if seq != None: return [d] + (["food"] if (nx, ny) in Global.food else []) + seq
                        occupied.pop()
                        foodTaken -= (nx, ny) in Global.food
                    return None
                seq = explore(playerIndex, Global.geese[playerIndex][0], [], 0, 0)
                if seq != None:
                    Logger.log(f"Decision: chase tail, seq={seq}")
                    return seq[0]
            if mustEat: return None
        return None

    def chaseSelf2(playerIndex, heatMap, coolMap): # NOTE: allows gap of size 1
        return None
        if len(Global.geese[playerIndex]) < 4: return None
        for maxSteps in range(1, Consts.MAX_CHASE_STEPS):
            def explore(playerIndex, xy, occupied, step, foodTaken):
                if step == maxSteps: return None
                for d in range(4):
                    nx, ny = Helper.moveTowards(xy, d)
                    if (nx, ny) in occupied: continue # my new steps are colliding
                    if heatMap[nx, ny] + foodTaken * ((nx, ny) in Global.geese[playerIndex]) > step+1: continue
                    if coolMap[nx, ny] <= step+1: continue
                    shrinks = Global.roundCnt % 40 > (Global.roundCnt + step) % 40
                    if step == heatMap[nx, ny]+foodTaken-shrinks and (nx, ny) in Global.geese[playerIndex]:
                        return [d]
                    foodTaken += (nx, ny) in Global.food
                    occupied.append((nx, ny))
                    seq = explore(playerIndex, (nx, ny), occupied, step+1, foodTaken)
                    if seq != None: return [d] + (["food"] if (nx, ny) in Global.food else []) + seq
                    occupied.pop()
                    foodTaken -= (nx, ny) in Global.food
                return None
            seq = explore(playerIndex, Global.geese[playerIndex][0], [], 0, 0)
            if seq != None:
                Logger.log(f"Decision: chase tail gap=1, seq={seq}")
                return seq[0]
        return None

    def maximizeSteps(playerIndex, heatMap, coolMap, safe, threshold): # TODO: with tolerableHeatDis?
        def explore(playerIndex, xy, occupied, step, foodTaken):
            best = (0, [])
            if step >= Consts.MAX_SAFE_STEPS: return best
            for d in range(4):
                nx, ny = Helper.moveTowards(xy, d)
                if (nx, ny) in occupied: continue # my new steps are colliding
                shrinks = Global.roundCnt % 40 > (Global.roundCnt + step) % 40
                if heatMap[nx, ny] + (foodTaken-shrinks) * ((nx, ny) in Global.geese[playerIndex]) > step+1: continue
                if coolMap[nx, ny] <= step+1+(not safe): continue
                foodTaken += (nx, ny) in Global.food
                occupied.append((nx, ny))
                res = explore(playerIndex, (nx, ny), occupied, step+1, foodTaken)
                if res[0]+1 > best[0]: best = (res[0]+1, [d]+res[1])
                occupied.pop()
                foodTaken -= (nx, ny) in Global.food
            return best
        num, seq = explore(playerIndex, Global.geese[playerIndex][0], [], 0, 0)
        assert num == len(seq)
        Logger.log(f"Maximized safe={safe}, threshold={threshold}: {seq}")
        if seq != None and num >= threshold:
            Logger.log(f"Decision: maximize safe steps, seq={seq}")
            return seq[0]
        return None

    def chaseAdjHead(playerIndex, heatMap, coolMap, minHeatDisGuaranteed):
        for h in range(11, 0, -1):
            for d in range(4):
                if (d^2) == Global.prevMoves[playerIndex]: continue
                x, y = Helper.moveTowards(Global.geese[playerIndex][0], d)
                if minHeatDisGuaranteed[d] < h or coolMap[x, y] != 1: continue
                return Logger.log(f"Decision: chase adj head", d)

    def chaseAnyTailUpToDis(playerIndex, heatMap, coolMap, disFromPlayers, tolerableTailTurnsRange, dis):
        assert dis <= 2

        for tolerableTailTurns in tolerableTailTurnsRange:
            for d in range(4):
                if (d^2) == Global.prevMoves[playerIndex]: continue
                x, y = Helper.moveTowards(Global.geese[playerIndex][0], d)
                for g in range(4):
                    if len(Global.geese[g]) == 0: continue
                    if g != playerIndex: # if it's my tail, no worries
                        if heatMap[x, y] > 1 or coolMap[x, y] <= 1: continue
                        if any(disFromPlayers[(g,)+f] < tolerableTailTurns for f in Global.food): continue
                    if (x, y) == Global.geese[g][-1]: return Logger.log(f"Decision: chase player {g}'s tail, tolerableTailTurns={tolerableTailTurns}", d)

            if dis > 1:
                for d in range(4):
                    if (d^2) == Global.prevMoves[playerIndex]: continue
                    x, y = Helper.moveTowards(Global.geese[playerIndex][0], d)
                    if heatMap[x, y] > 1 or coolMap[x, y] <= 1: continue
                    for q in range(4):
                        nx, ny = Helper.moveTowards((x, y), q)
                        if coolMap[nx, ny] <= 1: continue
                        for g in range(4):
                            if len(Global.geese[g]) == 0: continue
                            if g != playerIndex: # if it's my tail, no worries
                                if heatMap[x, y] > 2 or coolMap[x, y] <= 2: continue
                                if any(disFromPlayers[(g,)+f] < tolerableTailTurns+1 for f in Global.food): continue
                            if (nx, ny) == Global.geese[g][-1]: return Logger.log(f"Decision: chase player {g}'s tail, tolerableTailTurns={tolerableTailTurns}", d)

    def enumHeatFixCoolStridePreventHoneyPot(playerIndex, minDisGuaranteed, minHeatDisGuaranteed, heatRange, coolThreshold, strideThreshold):
        assert coolThreshold > 1
        for heatThreshold in heatRange:
            for d in range(4):
                if (d^2) == Global.prevMoves[playerIndex]: continue
                if minDisGuaranteed[coolThreshold, d] < strideThreshold: continue
                if minDisGuaranteed[1, d] == minDisGuaranteed[coolThreshold, d]: continue
                if minHeatDisGuaranteed[d] < heatThreshold: continue
                return Logger.log(f"Decision: no food no honeypot with heatDis={minHeatDisGuaranteed[d]}, cool={coolThreshold}, stride={minDisGuaranteed[coolThreshold, d]}", d)

    def enumHeatFixCoolStride(playerIndex, minDisGuaranteed, minHeatDisGuaranteed, heatRange, coolThreshold, strideThreshold):
        for heatThreshold in heatRange:
            for d in range(4):
                if (d^2) == Global.prevMoves[playerIndex]: continue
                if minDisGuaranteed[coolThreshold, d] < strideThreshold: continue
                if minHeatDisGuaranteed[d] < heatThreshold: continue
                return Logger.log(f"Decision: no food with heatDis={minHeatDisGuaranteed[d]}, cool={coolThreshold}, stride={minDisGuaranteed[coolThreshold, d]}", d)

    def randomFallback(playerIndex):
        d = [i for i in range(4) if (i^2) != Global.prevMoves[playerIndex]][random.randint(0, 2)]
        return Logger.log(f"Decision: random", d)

class Conditions:
    def tryToChaseMyselfCozImTooLong(playerIndex):
        return len(Global.geese[playerIndex]) >= Consts.MAX_LENGTH and len(Global.geese[playerIndex]) > max(len(Global.geese[i]) for i in range(4) if i != playerIndex) + 2

    def tryToChaseMyselfCozBoardTooSqueezy(playerIndex):
        return sum(len(Global.geese[i]) for i in range(4)) >= Consts.MAX_BOARD_SIZE and sum(len(Global.geese[i]) > 0 for i in range(4)) > 2

    def tryToHeadShotCozImLonger(playerIndex):
        return False

def getStep(playerIndex): # maybe you want to put somebody else's?
    for oppBold in [True, False]:
        Logger.log(f"oppBold={oppBold}")

        heatMap = Maps.generateHeatMap()
        disFromPlayers = Maps.generateDisFromPlayers(heatMap, oppBold)
        coolMap = Maps.generateCoolMap(playerIndex, disFromPlayers)
        heatMapOpp = Maps.generateOppHeatMapConsideringFood(disFromPlayers)

        # upto here no consideration for food

        # TODO: dis from food
        Global.food.sort(key=lambda p: disFromPlayers[(playerIndex,)+p] - min(disFromPlayers[(i,)+p] for i in range(4) if i != playerIndex))
        heatMap = Maps.generateHeatMapConsideringFood(disFromPlayers)
        disFromPlayers = Maps.generateDisFromPlayers(heatMap, oppBold)
        coolMap = Maps.generateCoolMap(playerIndex, disFromPlayers)

        Global.food.sort(key=lambda p: disFromPlayers[(playerIndex,)+p] - min(disFromPlayers[(i,)+p] for i in range(4) if i != playerIndex))
        minHeatDisGuaranteed = Maps.generateMinHeatDisGuaranteed(playerIndex, heatMap, coolMap)
        minDisGuaranteed = Maps.generateMinDisGuaranteed(playerIndex, heatMap, coolMap)

        # interesting...
        # step = Decisions.maximizeSteps(playerIndex, heatMapOpp, coolMap, True, 6)
        # if step != None: return step

        if Conditions.tryToChaseMyselfCozImTooLong(playerIndex):
            step = Decisions.chaseSelf(playerIndex, heatMapOpp, coolMap)
            if step != None: return step
            step = Decisions.chaseSelf2(playerIndex, heatMapOpp, coolMap)
            if step != None: return step

        if Conditions.tryToChaseMyselfCozBoardTooSqueezy(playerIndex):
            Logger.log(f"Condition: board is squeezy")
            step = Decisions.chaseSelf(playerIndex, heatMapOpp, coolMap)
            if step != None: return step
            step = Decisions.chaseSelf2(playerIndex, heatMapOpp, coolMap)
            if step != None: return step

        step = Decisions.chaseSelf(playerIndex, heatMapOpp, coolMap, oppBold) # not oppBold => dangerous => not mustEat
        if step != None: return step

        if oppBold:
            Logger.log(f"heatMap:\n{heatMap}")
            Logger.log(f"coolMap:\n{coolMap}")
            Logger.log(f"minHeatDisGuaranteed:\n{minHeatDisGuaranteed}")
            Logger.log(f"minDisGuaranteed:\n{minDisGuaranteed}")
        # maps are adjusted to food

        # TODO: use maximizeSteps() to ensure that it's not a honeypot?
        for tolerableHeatDis in range(9, 6, -1):
            for it in range(3):
                if it==1 and len(Global.geese[Global.playerIndex]) >= Consts.LENGTH_AGGRESSIVE: break # TODO
                for f in Global.food:
                    # if disFromPlayers[(playerIndex,)+f] >= 8: continue
                    if disFromPlayers[(playerIndex,)+f] - min(disFromPlayers[(i,)+f] for i in range(4) if i != playerIndex) > 2: continue
                    if disFromPlayers[(playerIndex,)+f] - min(disFromPlayers[(i,)+f] for i in range(4) if i != playerIndex) == 2:
                        if min(disFromPlayers[(i,)+f] for i in range(4) if i != playerIndex) < 5: continue
                    for d in range(4):
                        if (d^2) == Global.prevMoves[playerIndex]: continue
                        nx, ny = Helper.moveTowards(Global.geese[playerIndex][0], d)
                        if heatMap[nx, ny] > 1 or coolMap[nx, ny] <= 1: continue
                        if minHeatDisGuaranteed[d] < tolerableHeatDis: continue

                        tolerableStride = 6 if disFromPlayers[(playerIndex,)+f] < 3 else 8
                        if it == 0:
                            if Helper.getManhattanDis((nx, ny), f) >= Helper.getManhattanDis(Global.geese[playerIndex][0], f): continue
                        if it >= 1:
                            tolerableStride = 5 if disFromPlayers[(playerIndex,)+f] < 5 else 7
                            if Helper.getManhattanDis((nx, ny), f) > Helper.getManhattanDis(Global.geese[playerIndex][0], f): continue
                        if it == 2:
                            tolerableStride = 4 if disFromPlayers[(playerIndex,)+f] < 4 else 6
                            if len(Global.geese[playerIndex]) == min(len(Global.geese[i]) for i in range(4) if len(Global.geese[i]) > 0):
                                tolerableStride -= 1
                            # if (nx, ny) in Global.food:
                            #     if minDisGuaranteed[2, d] > tolerableHeatDis and heatMap[nx, ny] < 2: return Logger.log(f"Decision: chase adj food {f} at dis {disFromPlayers[(playerIndex,)+f]} with heatDis={minHeatDisGuaranteed[d]}", d)

                        for tolerableCoolThreshold in range(Consts.MAX_COOL_THRESHOLD, 4-(it==2), -1):
                            for stride in range(9, tolerableStride-1, -1):
                                for coolThreshold in range(Consts.MAX_COOL_THRESHOLD, tolerableCoolThreshold-1, -1):
                                    if minDisGuaranteed[1, d] == minDisGuaranteed[coolThreshold, d]: continue
                                    if minDisGuaranteed[coolThreshold, d] >= stride: return Logger.log(f"Decision: chase food {f} at dis {disFromPlayers[(playerIndex,)+f]} with heatDis={minHeatDisGuaranteed[d]}, cool={coolThreshold}, stride={stride}", d)

                        for tolerableCoolThreshold in range(Consts.MAX_COOL_THRESHOLD, 3-(it==2), -1):
                            for stride in range(9, tolerableStride+2-1, -1):
                                for coolThreshold in range(Consts.MAX_COOL_THRESHOLD, tolerableCoolThreshold-1, -1):
                                    if minDisGuaranteed[1, d] == minDisGuaranteed[coolThreshold, d]: continue
                                    if minDisGuaranteed[coolThreshold, d] >= stride: return Logger.log(f"Decision: chase food {f} at dis {disFromPlayers[(playerIndex,)+f]} with heatDis={minHeatDisGuaranteed[d]}, cool={coolThreshold}, stride={stride}", d)
                    if it == 0: break # tier 1 is closest

        step = Decisions.chaseSelf(playerIndex, heatMapOpp, coolMap)
        if step != None: return step

        for cool in range(Consts.MAX_COOL_THRESHOLD, 2, -1):
            for stride in range(11, 8, -1):
                step = Decisions.enumHeatFixCoolStridePreventHoneyPot(playerIndex, minDisGuaranteed, minHeatDisGuaranteed, range(9, 7, -1), cool, stride)
                if step != None: return step

        for cool, stride in [[5, 8], [4, 10], [3, 10], [4, 9], [4, 8], [3, 9], [5, 7], [4, 9], [4, 7], [3, 8], [3, 7]]:
            step = Decisions.enumHeatFixCoolStridePreventHoneyPot(playerIndex, minDisGuaranteed, minHeatDisGuaranteed, range(11, 6, -1), cool, stride)
            if step != None: return step

        for cool, stride in [[5, 8], [4, 10], [3, 10], [4, 9], [3, 9], [4, 8], [4, 7], [3, 8], [2, 10], [4, 6], [3, 7], [3, 6]]:
            step = Decisions.enumHeatFixCoolStridePreventHoneyPot(playerIndex, minDisGuaranteed, minHeatDisGuaranteed, range(11, 6, -1), cool, stride)
            if step != None: return step

        if oppBold: continue

        step = Decisions.maximizeSteps(playerIndex, heatMapOpp, coolMap, True, 6)
        if step != None: return step

        for cool, stride in [[5, 8], [4, 10], [3, 10], [4, 9], [3, 9], [4, 8], [4, 7], [3, 8], [2, 10], [4, 6], [3, 7], [3, 6], [2, 9], [2, 8]]:
            step = Decisions.enumHeatFixCoolStride(playerIndex, minDisGuaranteed, minHeatDisGuaranteed, range(11, 6, -1), cool, stride)
            if step != None: return step

        step = Decisions.maximizeSteps(playerIndex, heatMapOpp, coolMap, False, 6)
        if step != None: return step

        for cool, stride in [[4, 7], [3, 7], [3, 6], [2, 8], [2, 7], [2, 6], [3, 5], [3, 4], [1, 9], [1, 7], [2, 3], [0, 8], [1, 6], [0, 7], [0, 3]]:
            step = Decisions.enumHeatFixCoolStride(playerIndex, minDisGuaranteed, minHeatDisGuaranteed, range(11, 3, -1), cool, stride)
            if step != None: return step

        step = Decisions.chaseSelf2(playerIndex, heatMapOpp, coolMap)
        if step != None: return step

    step = Decisions.maximizeSteps(playerIndex, heatMapOpp, coolMap, False, 1)
    if step != None: return step

    step = Decisions.chaseAdjHead(playerIndex, heatMap, coolMap, minHeatDisGuaranteed)
    if step != None: return step

    return Decisions.randomFallback(playerIndex)

def agent(obs_dict, config_dict):
    Global.update(Observation(obs_dict), Configuration(config_dict))
    return Consts.CODE[getStep(Global.playerIndex)]

