from manim import *

class BeamReflectionCompact(Scene):
    def construct(self):
        # Create the blue box (mirrors) at the top half of the screen
        top_mirror = Line(
            start=LEFT * 5, 
            end=RIGHT * 5, 
            color=BLUE, 
            stroke_width=4
        ).shift(UP * 1.5)
        
        bottom_mirror = Line(
            start=LEFT * 5, 
            end=RIGHT * 5, 
            color=BLUE, 
            stroke_width=4
        ).shift(UP * 0.5)

        top_mirror_name = MathTex(r"r_{i}", color = BLUE)
        top_mirror_name.scale(0.7)
        top_mirror_name.next_to(top_mirror, LEFT, buff=0.5)
        bottom_mirror_name = MathTex(r"r_{e}", color = BLUE)
        bottom_mirror_name.scale(0.7)
        bottom_mirror_name.next_to(bottom_mirror, LEFT, buff=0.5)

        self.play(Create(top_mirror), Create(bottom_mirror),Write(top_mirror_name), Write(bottom_mirror_name))
        
        # Define the starting point for the beam
        beam_start = LEFT * 4.5 + UP * 2
        beam_end = LEFT * 3.5 + UP * 1.5
        
        # Create the incoming red beam
        incoming_beam = Line(
            start=beam_start,
            end=beam_end,
            color=RED,
            stroke_width=2,
        )
        self.play(Create(incoming_beam), run_time=0.5)

        # Animate multiple reflections (8 reflections)
        current_start = beam_end
        reflection_lines = []
        direction = -1  # -1 for downward, +1 for upward

        for _ in range(8):
            next_end = current_start + RIGHT * 1 + UP * direction * 1
            reflection = Line(
                start=current_start, 
                end=next_end, 
                color=RED, 
                stroke_width=2
            )
            reflection_lines.append(reflection)
            current_start = next_end
            direction *= -1  # Alternate the direction
            self.play(Create(reflection), run_time=0.5)

        # Add the outgoing beam
        outgoing_beam = Line(
            start=current_start,
            end=current_start + RIGHT * 1.5 + UP * direction * 1.5,
            color=RED,
            stroke_width=2,
        )
        self.play(Create(outgoing_beam), run_time=0.5)
        self.wait(2)

        distance = MathTex("D_{eff}=","N",".L").next_to(bottom_mirror, DOWN, buff=1)
        self.play(Write(distance))
        self.wait(2)

        self.play(distance[1].animate.set_color(YELLOW))  # Flash color
        self.wait(1)  # Hold the flash
        

        N = MathTex(r"N=\frac{\mathcal{F}}{2.\pi}")
        F = MathTex(r"\mathcal{F}=\pi\frac{\sqrt{R}}{1-R}")
        F.move_to(DOWN*2+RIGHT*2)
        N.move_to(DOWN*2+LEFT*2)
        self.play(Write(N))
        self.wait(1)
        self.play(Write(F))
        self.wait(3)
        V = MathTex(r"\text{VIRGO}(\mathcal{F}=50)",r"\quad","D_{eff}=24Km").to_edge(DOWN)
        Advanced_V = MathTex(r"\text{Advanced VIRGO}(\mathcal{F}=450)",r"\quad","D_{eff}=216Km").to_edge(DOWN)
        self.play(Write(V))
        self.wait(2)
        self.play(FadeOut(V))
        self.play(Write(Advanced_V))
        self.wait(3)

        