from manim import *

class Axes3D(ThreeDScene):
    def Create_axes(self):
        # Create 3D axes
        axes = ThreeDAxes()
        x_label = axes.get_x_axis_label(Tex("x"))
        y_label = axes.get_y_axis_label(Tex("y")).shift(UP * 1.8)

        # Create a grid in the XY plane
        grid = NumberPlane()

        # Create dots simulating 2 astrophysique masses, initially positioned along X and Y axes.
        mass1 = Sphere(radius=0.2, color=BLUE).shift(RIGHT)  # Dot on the X-axis
        mass2 = Sphere(radius=0.2, color=BLUE).shift(LEFT)   # Dot on the Y-axis

        # Set up the camera with zoom out effect
        self.set_camera_orientation(zoom=0.5)

        # Animation: Fade in the axes, the grid, and the dots
        self.play(FadeIn(axes), FadeIn(x_label), FadeIn(y_label))
        self.play(FadeIn(grid))
        self.play(FadeIn(mass1), FadeIn(mass2))
       
        self.wait(1)

        # Zoom out the camera to see the whole axes
        self.move_camera(phi=75 * DEGREES, theta=30 * DEGREES, zoom=1, run_time=1.5)

        self.wait(2)

        # Rotate the spheres around the Z-axis multiple times
        self.play(Rotate(mass1, angle=10*PI, axis=OUT, about_point=ORIGIN, run_time=10),
                  Rotate(mass2, angle=10*PI, axis=OUT, about_point=ORIGIN, run_time=10))

        self.wait(2)

    def construct(self):
        # Call the axes creation function
        self.Create_axes()
