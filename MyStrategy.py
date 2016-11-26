import math
import random
import sys

from model.ActionType import ActionType
from model.Faction import Faction
from model.Game import Game
from model.Move import Move
from model.Wizard import Wizard
from model.World import World
from model.LaneType import LaneType
from model.Bonus import Bonus
from model.BonusType import BonusType
from model.Tree import Tree

WAYPOINT_RADIUS = 100.00
LOW_HP_FACTOR = 0.25
MAP_SIZE = 4000

class Vector(object):
    def __init__(self, *args):
        """ Create a vector, example: v = Vector(1,2) """
        if len(args)==0: self.values = (0,0)
        else: self.values = args
        
    def norm(self):
        """ Returns the norm (length, magnitude) of the vector """
        return math.sqrt(sum( comp**2 for comp in self ))
        
    def argument(self):
        """ Returns the argument of the vector, the angle clockwise from +y."""
        arg_in_rad = math.acos(Vector(0,1)*self/self.norm())
        arg_in_deg = math.degrees(arg_in_rad)
        if self.values[0]<0: return 360 - arg_in_deg
        else: return arg_in_deg

    def normalize(self):
        """ Returns a normalized unit vector """
        norm = self.norm()
        normed = tuple( comp/norm for comp in self )
        return Vector(*normed)
    
    def rotate(self, *args):
        """ Rotate this vector. If passed a number, assumes this is a 
            2D vector and rotates by the passed value in degrees.  Otherwise,
            assumes the passed value is a list acting as a matrix which rotates the vector.
        """
        if len(args)==1 and type(args[0]) == type(1) or type(args[0]) == type(1.):
            # So, if rotate is passed an int or a float...
            if len(self) != 2:
                raise ValueError("Rotation axis not defined for greater than 2D vector")
            return self._rotate2D(*args)
        elif len(args)==1:
            matrix = args[0]
            if not all(len(row) == len(matrix) for row in matrix) or not len(matrix)==len(self):
                raise ValueError("Rotation matrix must be square and same dimensions as vector")
            return self.matrix_mult(matrix)
        
    def _rotate2D(self, theta):
        """ Rotate this vector by theta in degrees.
            
            Returns a new vector.
        """
        theta = math.radians(theta)
        # Just applying the 2D rotation matrix
        dc, ds = math.cos(theta), math.sin(theta)
        x, y = self.values
        x, y = dc*x - ds*y, ds*x + dc*y
        return Vector(x, y)
        
    def matrix_mult(self, matrix):
        """ Multiply this vector by a matrix.  Assuming matrix is a list of lists.
        
            Example:
            mat = [[1,2,3],[-1,0,1],[3,4,5]]
            Vector(1,2,3).matrix_mult(mat) ->  (14, 2, 26)
         
        """
        if not all(len(row) == len(self) for row in matrix):
            raise ValueError('Matrix must match vector dimensions') 
        
        # Grab a row from the matrix, make it a Vector, take the dot product, 
        # and store it as the first component
        product = tuple(Vector(*row)*self for row in matrix)
        
        return Vector(*product)
    
    def inner(self, other):
        """ Returns the dot product (inner product) of self and other vector
        """
        return sum(a * b for a, b in zip(self, other))
    
    def x(self):
        return self.values[0]

    def y(self):
        return self.values[1]

    def get_distance_to(self, _x, _y):
        return math.hypot(self.x() - _x, self.y() - _y)

    def get_distance_to_point(self, point):
        return self.get_distance_to(point.x(), point.y())

    def get_distance_to_unit(self, unit):
        return self.get_distance_to(unit.x, unit.y)

    def __mul__(self, other):
        """ Returns the dot product of self and other if multiplied
            by another Vector.  If multiplied by an int or float,
            multiplies each component by other.
        """
        if type(other) == type(self):
            return self.inner(other)
        elif type(other) == type(1) or type(other) == type(1.0):
            product = tuple( a * other for a in self )
            return Vector(*product)
    
    def __rmul__(self, other):
        """ Called if 4*self for instance """
        return self.__mul__(other)
            
    def __div__(self, other):
        if type(other) == type(1) or type(other) == type(1.0):
            divided = tuple( a / other for a in self )
            return Vector(*divided)
    
    def __add__(self, other):
        """ Returns the vector addition of self and other """
        added = tuple( a + b for a, b in zip(self, other) )
        return Vector(*added)
    
    def __sub__(self, other):
        """ Returns the vector difference of self and other """
        subbed = tuple( a - b for a, b in zip(self, other) )
        return Vector(*subbed)
    
    def __iter__(self):
        return self.values.__iter__()
    
    def __len__(self):
        return len(self.values)
    
    def __getitem__(self, key):
        return self.values[key]
        
    def __repr__(self):
        return str(self.values)

BUILDING_DISTANCE_FACTOR = 1.0
WIZARD_DISTANCE_FACTOR = 0.7
MINION_DISTANCE_FACTOR = 1.0
CRATE_POINTS = (Vector(1202, 1202), Vector(2798, 2798))

def closest_point_on_seg(a : Vector, b : Vector, pt : Vector):
    seg_v = b - a
    pt_v = pt - a
    seg_len = seg_v.norm()
    seg_v_unit = seg_v.normalize()
	
    proj = pt_v * seg_v_unit
    if proj <= 0:
        return a
    if proj >= seg_len:
        return b
	
    return a + (seg_v_unit * proj) 

def is_segment_circle_intersect(a : Vector, b : Vector, circle_center : Vector, circle_radius : float):
    p = closest_point_on_seg(a, b, circle_center)
    return (p - circle_center).norm() < circle_radius

def get_targets(world: World):
    for b in world.buildings:
        yield (b, BUILDING_DISTANCE_FACTOR)
    for w in world.wizards:
        if not w.me:
            yield (w, WIZARD_DISTANCE_FACTOR)
    for m in world.minions:
        yield (m, MINION_DISTANCE_FACTOR)

def get_nearest_target(me: Wizard, world: World, targetPoint):
    # tree special check
    for t in world.trees:
        distance = me.get_distance_to_unit(t)
        angle = abs(me.get_angle_to(targetPoint[0], targetPoint[1]))
        if distance < (me.radius + t.radius) * 1.2 and angle < math.pi * 0.25:
            return t

    nearest_target = None
    nearest_target_distance = sys.float_info.max

    for target, factor in get_targets(world):
        # Нейтралов атакуем тоже если их хп меньше максимального - они стригеренны
        if (target.faction == me.faction or
                        target.faction == Faction.NEUTRAL and target.life >= target.max_life):
            continue

        distance = me.get_distance_to_unit(target) * factor
        if distance < nearest_target_distance:
            nearest_target = target
            nearest_target_distance = distance
    return nearest_target

def get_nearest_and_dist(me: Wizard, points):
    nearest_target = None
    nearest_target_distance = sys.float_info.max

    for c in points:
        # Нейтралов атакуем тоже если их хп меньше максимального - они стригеренны
        distance = me.get_distance_to(c[0], c[1])
        if distance < nearest_target_distance:
            nearest_target = c
            nearest_target_distance = distance
    return nearest_target, nearest_target_distance

def get_nearest_world_crate_and_dist(me: Wizard, world: World):
    nearest_target = None
    nearest_target_distance = sys.float_info.max

    for c in world.bonuses:
        # Нейтралов атакуем тоже если их хп меньше максимального - они стригеренны
        distance = me.get_distance_to(c.x, c.y)
        if distance < nearest_target_distance:
            nearest_target = Vector(c.x, c.y)
            nearest_target_distance = distance
    return nearest_target, nearest_target_distance


def get_next_waypoint(waypoints, me: Wizard, world: World, game: Game):
    """
    Данный метод предполагает, что все ключевые точки на линии упорядочены по уменьшению дистанции до последней
    ключевой точки. Перебирая их по порядку, находим первую попавшуюся точку, которая находится ближе к последней
    точке на линии, чем волшебник. Это и будет следующей ключевой точкой.

    Дополнительно проверяем, не находится ли волшебник достаточно близко к какой-либо из ключевых точек. Если это
    так, то мы сразу возвращаем следующую ключевую точку.
    """

    last_waypoint = waypoints[-1]
    for i, waypoint in enumerate(waypoints[:-1]):
        if waypoint.get_distance_to_unit(me) <= WAYPOINT_RADIUS:
            return waypoints[i + 1]

        if last_waypoint.get_distance_to_point(waypoint) < last_waypoint.get_distance_to_unit(me):
            return waypoint

    return last_waypoint


def get_lane(me: Wizard, game: Game):
    if me.id in [1, 2, 6, 7]:
        return LaneType.TOP
    elif me.id in [3, 8]:
        return LaneType.MIDDLE
    else:
        return LaneType.BOTTOM

def get_lane_center_point(lane : LaneType):
    if lane == LaneType.TOP:
        return Vector(200.0, 200.0)
    elif lane == LaneType.MIDDLE:
        return Vector(2000.0, 2000.0)
    else:
        return Vector(MAP_SIZE - 200.0, MAP_SIZE - 200.0)
    

def get_waypoints(me: Wizard, game: Game):
    map_size = game.map_size
    lane = get_lane(me, game)
    if lane == LaneType.TOP:
        # точки верхней линии
        return [
            Vector(100.0, map_size - 100.0),
            Vector(100.0, map_size - 400.0),
            Vector(200.0, map_size - 800.0),
            Vector(200.0, map_size * 0.75),
            Vector(200.0, map_size * 0.5),
            Vector(200.0, map_size * 0.25),
            Vector(200.0, 200.0),
            Vector(map_size * 0.25, 200.0),
            Vector(map_size * 0.5, 200.0),
            Vector(map_size * 0.75, 200.0),
            Vector(map_size - 200.0, 200.0)
        ]
    elif lane == LaneType.MIDDLE:
        # точки средней линии
        return [
            Vector(100.0, map_size - 100.0),
            random.choice([Vector(600.0, map_size - 200.0), Vector(200.0, map_size - 600.0)]),
            Vector(800.0, map_size - 800.0),
            Vector(map_size - 600.0, 600.0)
        ]
    else:
        # точки нижней линии
        return [
            Vector(100.0, map_size - 100.0),
            Vector(400.0, map_size - 100.0),
            Vector(800.0, map_size - 200.0),
            Vector(map_size * 0.25, map_size - 200.0),
            Vector(map_size * 0.5, map_size - 200.0),
            Vector(map_size * 0.75, map_size - 200.0),
            Vector(map_size - 200.0, map_size - 200.0),
            Vector(map_size - 200.0, map_size * 0.75),
            Vector(map_size - 200.0, map_size * 0.5),
            Vector(map_size - 200.0, map_size * 0.25),
            Vector(map_size - 200.0, 200.0)
        ]

def lerp(a : float, b : float, t : float):
    return a + (b - a) * t

def lerp_clamped(a : float, b : float, t : float):
    return clamp(lerp(a, b, t), min(a, b), max(a,b))

def clamp(x : float, v0 : float, v1 : float):
    return max(min(x, v1), v0)

class MyStrategy:

    def __init__(self):
        super().__init__()
        self.initialized = False
        self.waypoints = None

    def initialize(self, me: Wizard, game: Game):
        random.seed(game.random_seed)
        self.waypoints = get_waypoints(me, game)
        self.initialized = True

    def apply_move(self, target: Vector, lookAt: Vector, me: Wizard, world: World, game: Game, move: Move):
        lookAngle = me.get_angle_to(lookAt.x(), lookAt.y())
        move.turn = lookAngle
        
        moveDir = target - Vector(me.x, me.y)
        moveDir = moveDir.normalize()

        # repulse
        targets = []
        targets.extend(world.buildings)
        targets.extend([w for w in world.wizards if not w.me])
        targets.extend(world.minions)
        targets.extend(world.trees)

        REPULSE_RADIUS_FACTOR = 1.5
        REPULSE_DISTANCE = 30.0
        REPULSE_RANDOM = 0.1

        t = world.tick_index * 0.05
        random_repulse = Vector(math.sin(t), math.cos(t)).normalize() * REPULSE_RANDOM

        for target in targets:
            distance = me.get_distance_to_unit(target)
            radius = me.radius + target.radius
            if distance < radius * REPULSE_RADIUS_FACTOR:
                factor = lerp_clamped(1.1, 0.0, abs(distance - radius) / REPULSE_DISTANCE)
                repulse = Vector(me.x - target.x, me.y - target.y).normalize()
                moveDir = moveDir + (repulse + random_repulse).normalize() * factor
                moveDir = moveDir.normalize()

        moveAngle = me.get_angle_to(me.x + moveDir.x(), me.y + moveDir.y())
        move.speed = math.cos(moveAngle) * game.wizard_forward_speed
        move.strafe_speed = math.sin(moveAngle) * game.wizard_strafe_speed

    def move(self, me: Wizard, world: World, game: Game, move: Move):
        if not self.initialized:
            self.initialize(me, game)

        mePoint = Vector(me.x, me.y)
        targetPoint = get_next_waypoint(self.waypoints, me, world, game)
        lookAt = targetPoint

        FIGHT_RANGE_FACTOR = 1.7
        LOW_HEALTH = 0.3
        HEALTH_DIST_FACTOR = me.cast_range

        # check me in center
        CENTER_DIST = 500 
        CENTER_POINTS = (Vector(1000, 1000), Vector(3000, 3000))
        point, dist = get_nearest_and_dist(me, CENTER_POINTS)
        if dist < CENTER_DIST:
            # in center!
            lane = get_lane(me, game)
            targetPoint = get_lane_center_point(lane)
            lookAt = targetPoint
            FIGHT_RANGE_FACTOR = 1.0
        

        # crate check
        CRATE_TICK = 2500
        DIAGONAL_SIZE = 650 # set to 0 to disable crates
        REACH_DIST = 300
        CRATE_DIST_TO_RUN = 2500
        if abs(me.x - me.y) < DIAGONAL_SIZE:
            crate, dist = get_nearest_world_crate_and_dist(me, world)
            if crate is not None and dist < CRATE_DIST_TO_RUN:
                targetPoint = crate
                lookAt = targetPoint
                FIGHT_RANGE_FACTOR = 1.0
            else:
                crate, dist = get_nearest_and_dist(me, CRATE_POINTS)
                next_crate_tick = (world.tick_index // CRATE_TICK + 1) * CRATE_TICK
                if next_crate_tick < world.tick_count:
                    reach_dist = (next_crate_tick - world.tick_index) * game.wizard_forward_speed
                    if abs(reach_dist - dist) < REACH_DIST:
                        targetPoint = crate  
                        lookAt = targetPoint
                        FIGHT_RANGE_FACTOR = 1.0

        # fight                        
        nearest_target = get_nearest_target(me, world, targetPoint)
        if nearest_target is not None:
            distance = me.get_distance_to_unit(nearest_target)
            if distance < me.cast_range * FIGHT_RANGE_FACTOR:
                lookAt = Vector(nearest_target.x, nearest_target.y)
                cooldown = me.remaining_cooldown_ticks_by_action[ActionType.MAGIC_MISSILE]
                health = me.life / me.max_life

                # check collide
                obstacle = False
                for target, _ in get_targets(world):
                    if target.faction == me.faction:
                        target_pos = Vector(target.x, target.y)
                        if is_segment_circle_intersect(mePoint, lookAt, target_pos, target.radius):
                            obstacle = True
                            break

                healthFactor = LOW_HEALTH * HEALTH_DIST_FACTOR / health if health < LOW_HEALTH else -2 * game.wizard_forward_speed
                desired_distance = me.cast_range + game.wizard_forward_speed * cooldown + healthFactor
                dirToTarget = (lookAt - mePoint).normalize()
                if distance > desired_distance:
                    targetPoint = mePoint + dirToTarget
                else:
                    targetPoint = get_next_waypoint(self.waypoints[::-1], me, world, game)
                    
                if obstacle:
                    sign = 1.0 if world.tick_index % 200 < 100 else -1.0
                    strafe_vec = dirToTarget.rotate(math.pi * 90 * sign)
                    targetPoint = mePoint + ((targetPoint - mePoint).normalize() + strafe_vec).normalize()

                move.min_cast_distance = max(0, distance - 50.0) if obstacle else 0.0 #50 to be sure, because things move

                if distance <= me.cast_range:
                    move.action = ActionType.MAGIC_MISSILE

        self.apply_move(targetPoint, lookAt, me, world, game, move)
