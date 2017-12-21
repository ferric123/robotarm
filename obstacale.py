import numpy as np
from rllab.misc.mako_utils import compute_rect_vertices
from rllab.envs.box2d.parser import find_body
from rllab.core.serializable import Serializable
from rllab.envs.box2d.box2d_env import Box2DEnv
from rllab.misc import autoargs
from rllab.misc.overrides import overrides

class ArmEnv(Box2DEnv, Serializable):
    @autoargs.inherit(Box2DEnv.__init__)
    def __init__(self, *args, **kwargs):

        #self.dt = .1    # refresh rate
        #self.on_goal = 0

        super(ArmEnv, self).__init__(
            self.model_path("arm.xml.mako"),
            *args, **kwargs
        )
        self.arm1 = find_body(self.world, "arm1")
        self.arm2 = find_body(self.world, "arm2")
        self.point = find_body(self.world, "point")
        Serializable.__init__(self, *args, **kwargs)


    @overides
    def reset(self):
        self._set_state(self.initial_state)
        self._invalidate_state_caches()
        bounds = np.array([
            [-np.pi/2,np.pi/2],
            [-np.pi/2,np.pi/2]
        ])
        low, high = bounds
        aarm1,aarm2 = np.random.uniform(low, high)
        self.arm1.angle = aarm1
        self.arm2.angle = aarm2
        return self.get_current_obs()


    @overides
    def compute_reward(self, action):
        yield
        vertices[0] = compute_rect_vertices()
        dist1 = [(0.1 - self.arm1.vertices[0]), (-0.5 - self.arm1.vertices[1])]
        dist2 = [(0.1 - self.arm2.vertices[0]), (-0.5 - self.arm2.vertices[1])]
        yield -np.sqrt(dist2[0]**2+dist2[1]**2)



    @overrides
    def is_current_done(self):
        return (0.1 - 0.05 < self.arm2.vertices[0] < 0.1 + 0.05) and \
               (-0.5 - 0.05 < self.arm2.vertices[1] < -0.5 + 0.05)


import math
    def judge_obstacale(self):
#obstacale
        x3 = self.ob1[0]
        y3 = self.ob1[1]

#arm2判定
        arm2_coord = compute_rect_vertices(self.arm2.position, self.get_tip_pos(), 0.05)
#line1根据a,b两点构造直线方程 AX+BY+C=0
        A1=arm2_coord[2][1]-arm2_coord[3][1] #y2-y1
        B1=arm2_coord[2][0]-arm2_coord[3][0] #x2-x1
        C1=arm2_coord[2][0]*arm2_coord[3][1]-arm2_coord[3][0]*arm2_coord[2][1]  #x2*y1-x1*y2
#计算obstacale点到直线距离
        D1=abs(A1*x3+B1*y3+C1)/math.sqrt(A1*A1+B1*B1)

#line2根据a,b两点构造直线方程 AX+BY+C=0
        A2=arm2_coord[1][1]-arm2_coord[0][1]
        B2=arm2_coord[1][0]-arm2_coord[0][0]
        C2=arm2_coord[1][0]*arm2_coord[0][1]-arm2_coord[0][0]*arm2_coord[1][1]
#计算obstacale点到直线距离
        D2=abs(A2*x3+B2*y3+C2)/math.sqrt(A2*A2+B2*B2)


#line3根据a,b两点构造直线方程 AX+BY+C=0
        A3=arm2_coord[1][1]-arm2_coord[2][1]
        B3=arm2_coord[1][0]-arm2_coord[2][0]
        C3=arm2_coord[1][0]*arm2_coord[2][1]-arm2_coord[2][0]*arm2_coord[1][1]
#计算obstacale点到直线距离
        D3=abs(A3*x3+B3*y3+C3)/math.sqrt(A3*A3+B3*B3)

#line2根据a,b两点构造直线方程 AX+BY+C=0
        A4=arm2_coord[0][1]-arm2_coord[3][1]
        B4=arm2_coord[0][0]-arm2_coord[3][0]
        C4=arm2_coord[0][0]*arm2_coord[3][1]-arm2_coord[3][0]*arm2_coord[0][1]
#计算obstacale点到直线距离
        D4=abs(A4*x3+B4*y3+C4)/math.sqrt(A4*A4+B4*B4)

#arm1判定
        arm1_coord = compute_rect_vertices(self.arm1.position, self.get_tip_pos(), 0.05)
#line1根据a,b两点构造直线方程 AX+BY+C=0
        A5=arm1_coord[2][1]-arm1_coord[3][1] #y2-y1
        B5=arm1_coord[2][0]-arm1_coord[3][0] #x2-x1
        C5=arm1_coord[2][0]*arm1_coord[3][1]-arm1_coord[3][0]*arm1_coord[2][1]  #x2*y1-x1*y2
#计算obstacale点到直线距离
        D5=abs(A5*x3+B5*y3+C5)/math.sqrt(A5*A5+B5*B5)

#line2根据a,b两点构造直线方程 AX+BY+C=0
        A6=arm1_coord[1][1]-arm1_coord[0][1]
        B6=arm1_coord[1][0]-arm1_coord[0][0]
        C6=arm1_coord[1][0]*arm1_coord[0][1]-arm1_coord[0][0]*arm1_coord[1][1]
#计算obstacale点到直线距离
        D6=abs(A6*x3+B6*y3+C6)/math.sqrt(A6*A6+B6*B6)


#line3根据a,b两点构造直线方程 AX+BY+C=0
        A7=arm1_coord[1][1]-arm1_coord[2][1]
        B7=arm1_coord[1][0]-arm1_coord[2][0]
        C7=arm1_coord[1][0]*arm1_coord[2][1]-arm1_coord[2][0]*arm1_coord[1][1]
#计算obstacale点到直线距离
        D7=abs(A7*x3+B7*y3+C7)/math.sqrt(A7*A7+B7*B7)

#line2根据a,b两点构造直线方程 AX+BY+C=0
        A8=arm1_coord[0][1]-arm1_coord[3][1]
        B8=arm1_coord[0][0]-arm1_coord[3][0]
        C8=arm1_coord[0][0]*arm1_coord[3][1]-arm1_coord[3][0]*arm1_coord[0][1]
#计算obstacale点到直线距离
        D8=abs(A8*x3+B8*y3+C8)/math.sqrt(A8*A8+B8*B8)

        return D1<0.2 and D2<0.2 and D5<0.2 and D6<0.2 and D3<1.1 and D4<1.1 and D7<1.1 and D8<1.1
