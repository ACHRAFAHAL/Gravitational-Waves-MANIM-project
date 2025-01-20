'''
gw_on_grid.py

Animations of a graviational wave effect on a set of test masses on a grid.

Craig Cahillane
Feb 15, 2021
'''

from manim import *
import numpy as np

class GWonGrid(Scene):
    def construct(self):

        # First, fade in a grid
        grid = NumberPlane()

        self.play(
            Create(grid)
        )
        self.wait()

        # Second, put test masses on the grid in a circle
        num_circles = 16
        big_circle_radius = 2
        circles = []
        for ii in np.arange(num_circles):
            xx = big_circle_radius * np.cos(2 * np.pi * ii / num_circles)
            yy = big_circle_radius * np.sin(2 * np.pi * ii / num_circles)

            circle = Circle()
            circle.set_fill(YELLOW, opacity=0.9)
            circle.set_stroke(YELLOW, width=1)
            circle.scale(0.1)
            circle.set_x(xx)
            circle.set_y(yy)

            # Create new attributes for circle, init_x and init_y
            circle.init_x = xx
            circle.init_y = yy

            circles.append(circle)

            self.play(Create(circle, run_time=0.15))
        self.wait()

        # circles_vgroup = Group(*circles)

        tex_matrix = MathTex(r"\begin{bmatrix} h_+ & h_\times \\  h_\times & -h_+      \end{bmatrix}")
        # color the tex_matrix
        # h_+ is yellow
        # h_c is red
        tex_matrix[0][1:3].set_color(YELLOW)
        tex_matrix[0][7:10].set_color(YELLOW)
        tex_matrix[0][3:7].set_color(RED)

        # Create GW strain variables that can be changed with animations
        h_plus = ValueTracker(0.0)
        h_cross = ValueTracker(0.0)
        phase = ValueTracker(0.0) # relative h_plus and h_cross phase

        # Create DecimalNumber labels, and tie them to the variables
        h_plus_num  = DecimalNumber(0.0, num_decimal_places=1, include_sign=True, color=YELLOW)
        h_cross_num = DecimalNumber(0.0, num_decimal_places=1, include_sign=True, color=RED)
        phi_num     = DecimalNumber(0.0, num_decimal_places=1, include_sign=True, color=GREEN)

        h_plus_num.add_updater(lambda d: d.set_value(h_plus.get_value()))
        h_cross_num.add_updater(lambda d: d.set_value(h_cross.get_value()))
        phi_num.add_updater(lambda d: d.set_value(phase.get_value()))

        # Make some TeX labels, and move the numbers next to the labels
        h_plus_tex = MathTex(r"h_+ = ", color=YELLOW)
        h_cross_tex = MathTex(r"h_\times = ", color=RED)
        phi_tex = MathTex(r"\phi = ", color=GREEN)
        h_plus_num.next_to(h_plus_tex)
        h_cross_num.next_to(h_cross_tex)
        phi_num.next_to(phi_tex)

        # Create Vector Groups holding the TeX together
        h_plus_group = VGroup(h_plus_tex, h_plus_num)
        h_cross_group = VGroup(h_cross_tex, h_cross_num)
        phi_group = VGroup(phi_tex, phi_num)

        # Arrange the TeX on screen
        tex_matrix.to_edge(RIGHT).to_edge(UP)
        h_plus_group.next_to(tex_matrix, DOWN)
        h_cross_group.next_to(h_plus_group, DOWN)
        phi_group.next_to(h_cross_group, DOWN)

        # Animate the TeX drawing
        self.play(Write(tex_matrix), run_time=2.0)
        self.play(Write(h_plus_group), run_time=1.0)
        self.play(Write(h_cross_group), run_time=1.0)
        
        self.wait()

        # Make some mirrors
        self.y_mirror = Rectangle(
            width=1.0, height=0.3, color=BLUE, fill_opacity=0.5
        ).shift(big_circle_radius * UP)
        self.y_mirror.init_x = 0.0
        self.y_mirror.init_y = 2.0

        self.x_mirror = Rectangle(
            width=0.3, height=1.0, color=BLUE, fill_opacity=0.5
        ).shift(big_circle_radius * RIGHT)
        self.x_mirror.init_x = 2.0
        self.x_mirror.init_y = 0.0

        # Strain Tex
        self.strain_value = MathTex(
            r"{{ h }} \approx 1 \times 10^{-21}"
        ).move_to(3*UP+4.7*LEFT)
        self.strain_value[0].set_color(YELLOW)

        # Animate the test masses on screen according to the 
        # incident gravitational wave
        self.now = 0
        self.freq = 2.4
        
        def time_updater(mob, dt):
            '''Single updater for time
            '''
            self.now = self.now + dt
            return 

        def circle_coordinate_updater(circle, dt):
            '''Updater function for the test mass coordinates.
            This one does not use delta time, but instead
            stores the initial location in the circle object
            and modulates relative to that.
            '''
            # self.now = self.now + dt
            cur_time = self.now

            h_p = h_plus.get_value()  * np.cos(self.freq * cur_time)
            h_c = h_cross.get_value() * np.cos(self.freq * cur_time + phase.get_value())

            matrix = np.array( [[h_p, h_c], [h_c, -h_p]] )
            init_vector = np.array([[circle.init_x], [circle.init_y]])

            new_vector = init_vector + np.dot(matrix, init_vector)

            new_x = new_vector[0, 0]
            new_y = new_vector[1, 0]

            circle.set_x( new_x )
            circle.set_y( new_y )
            return circle

        
        def grid_updater(grid):
            '''Updater function for the grid.  
            Just applies a matrix to the original grid each time
            '''
            cur_time = self.now

            h_p = h_plus.get_value()  * np.cos(self.freq * cur_time)
            h_c = h_cross.get_value() * np.cos(self.freq * cur_time + phase.get_value())

            new_grid = NumberPlane()
            matrix = np.array( [[1 + h_p, h_c], [h_c, 1 - h_p]] )
            new_grid.apply_matrix(matrix)

            grid = new_grid.copy()

            return grid

        for circle in circles:
            circle.add_updater(circle_coordinate_updater)

        self.y_mirror.add_updater(circle_coordinate_updater)
        self.x_mirror.add_updater(circle_coordinate_updater)

        # grid.add_updater(grid_updater)
        # self.add(grid)

        self.now = 0
        grid.add_updater(time_updater) # add time updater to the grid

        # pure plus polarization

        self.play(
            h_plus.animate.set_value(0.5),
            h_cross.animate.set_value(0.0),
            run_time=3.0
        )
        self.wait(6 * PI)

        # move back to zero, add one second pause
        self.play(
            h_plus.animate.set_value(0), 
            h_cross.animate.set_value(0),
            run_time=3.0
        )
        self.wait()

        # pure cross polarization
        for circle in circles:
            self.play(
                circle.animate.set_stroke(RED),
                run_time=0.025
            )
            self.play(
                circle.animate.set_fill(RED),
                run_time=0.025
            )

        self.now = 0
        self.play(
            h_plus.animate.set_value(0),
            h_cross.animate.set_value(0.5),
            run_time=3.0
        )
        self.wait(2 * PI)

        # move back to zero, add one second pause
        self.play(
            h_plus.animate.set_value(0), 
            h_cross.animate.set_value(0),
            run_time=3.0
        )
        self.wait()

        # general combo polarization
        self.play(Write(phi_group), run_time=1.0)

        for circle in circles:
            self.play(
                circle.animate.set_stroke(ORANGE),
                run_time=0.025
            )
            self.play(
                circle.animate.set_fill(ORANGE),
                run_time=0.025
            )

        self.now = 0
        self.play(
            h_plus.animate.set_value(0.5), 
            h_cross.animate.set_value(0.5),
            run_time=3.0
        )
        self.wait(2 * PI)

        # add in phase difference
        self.play(
            phase.animate.set_value( PI / 2 ), # relative h_plus and h_cross phase
            run_time=3.0,
            rate_func=linear,
        )
        self.wait(2 * PI)

        # add one second pause
        self.play(
            h_plus.animate.set_value(0), 
            h_cross.animate.set_value(0),
            run_time=3.0
        )
        self.wait()

        # general combo polarization
        self.now = 0
        self.play(
            phase.animate.set_value( 3 * PI / 5 ),
            h_plus.animate.set_value(0.7), 
            h_cross.animate.set_value(-0.3),
            run_time=3.0
        )
        self.wait(2 * PI)

        # add one second pause
        self.play(
            h_plus.animate.set_value(0), 
            h_cross.animate.set_value(0),
            run_time=3.0
        )
        self.wait(1)

        self.play(
            h_plus.animate.set_value(0.5), 
            h_cross.animate.set_value(0),
            run_time=3.0
        )
        self.wait(1)
    
        for ii, circle in enumerate(circles):
            if ii == 0:
                self.play(
                    Transform(circle, self.x_mirror),
                    run_time=0.15
                )
            elif ii == 4:
                self.play(
                    Transform(circle, self.y_mirror),
                    run_time=0.15
                )
            else:
                self.play(
                    Uncreate(circle),
                    run_time=0.15
                )
        self.wait(2 * PI)

        self.play( 
            h_plus.animate.set_value(0), 
            h_cross.animate.set_value(0),
            run_time=3.0
        )
        self.wait(1)

        self.play(
            Write(self.strain_value)
        )
        self.wait(1)