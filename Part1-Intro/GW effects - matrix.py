from manim import *

class GWMatrixExample(Scene):
    def construct(self):
        tex_matrix = MathTex(r"\begin{bmatrix} h_+ & h_\times \\  h_\times & -h_+ \end{bmatrix}")
        tex_matrix[0][1:3].set_color(YELLOW)
        tex_matrix[0][7:10].set_color(YELLOW)
        tex_matrix[0][3:7].set_color(RED)

        h_plus = ValueTracker(0.0)
        h_cross = ValueTracker(0.0)
        phase = ValueTracker(0.0)

        h_plus_num = DecimalNumber(0.0, num_decimal_places=1, include_sign=True, color=YELLOW)
        h_cross_num = DecimalNumber(0.0, num_decimal_places=1, include_sign=True, color=RED)
        phi_num = DecimalNumber(0.0, num_decimal_places=1, include_sign=True, color=GREEN)

        h_plus_num.add_updater(lambda d: d.set_value(h_plus.get_value()))
        h_cross_num.add_updater(lambda d: d.set_value(h_cross.get_value()))
        phi_num.add_updater(lambda d: d.set_value(phase.get_value()))

        h_plus_tex = MathTex(r"h_+ = ", color=YELLOW)
        h_cross_tex = MathTex(r"h_\times = ", color=RED)
        phi_tex = MathTex(r"\phi = ", color=GREEN)
        h_plus_num.next_to(h_plus_tex)
        h_cross_num.next_to(h_cross_tex)
        phi_num.next_to(phi_tex)

        h_plus_group = VGroup(h_plus_tex, h_plus_num)
        h_cross_group = VGroup(h_cross_tex, h_cross_num)
        phi_group = VGroup(phi_tex, phi_num)

        tex_matrix.to_edge(RIGHT).to_edge(UP)
        h_plus_group.next_to(tex_matrix, DOWN)
        h_cross_group.next_to(h_plus_group, DOWN)
        phi_group.next_to(h_cross_group, DOWN)

        self.play(Write(tex_matrix), run_time=2.0)
        self.play(Write(h_plus_group), run_time=1.0)
        self.play(Write(h_cross_group), run_time=1.0)
        self.wait()

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
            # self.now = self.now + dt
            cur_time = self.now

            h_p = h_plus.get_value()  * np.cos(self.freq * cur_time)
            h_c = h_cross.get_value() * np.cos(self.freq * cur_time + phase.get_value())

        def grid_updater(grid):
            pass  # Add your grid update logic here

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

        self.now = 0
        self.play(
            h_plus.animate.set_value(0.5), 
            h_cross.animate.set_value(0.5),
            run_time=3.0
        )
        self.wait(2 * PI)

        # add in phase difference
        self.play(
            phase.animate.set_value(PI / 2),  # relative h_plus and h_cross phase
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
            phase.animate.set_value(3 * PI / 5),
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
 