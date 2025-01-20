"""intro_to_ligo.py

Scene illustrating the beams and optics of Advanced LIGO
"""
 
from manim import *

def unrotate(x1, x2, y1, y2):
    """Returns the angle and (x1, y1), (x2, y2) coordinates for rotation of the electric field.
    (this is a horrible way to do this but rn idk)
    """

    # get mean for rotation
    x0 = (x1 + x2)/2
    y0 = (y1 + y2)/2

    delta_x = x2 - x1
    delta_y = y2 - y1

    theta = np.arctan2(delta_y, delta_x)
    length = np.sqrt(delta_x**2 + delta_y**2)

    new_x1 = x0 - length/2
    new_x2 = x0 + length/2
    new_y1 = y0
    new_y2 = y0

    return new_x1, new_x2, new_y1, new_y2, theta

class IntroToLIGO(Scene):
    def get_phase(self, k=1, x=0, freq=1, time=0, phase=0):
        return k * x + 2 * PI * freq * time + phase

    def get_sine_wave(self, amp=1, k=1, x=0, freq=1, time=0, phase=0):
        return  amp * np.sin(k * x + 2 * PI * freq * time + phase)

    def rotate(self, x1, y1, theta, origin=np.array([0,0,0])):
        """Return x2 and y2 being rotated about the origin by theta.
        """
        x0 = origin[0]
        y0 = origin[1]
        x2 = (x1 - x0) * np.cos(theta) - (y1 - y0) * np.sin(theta)
        y2 = (x1 - x0) * np.sin(theta) + (y1 - y0) * np.cos(theta)
        return x2, y2

    def make_electric_field(
        self, 
        amp=ValueTracker(1), 
        k=ValueTracker(1), 
        freq=ValueTracker(1), 
        time=0, 
        phase=ValueTracker(0), 
        start=np.array([-5,0,0]), 
        end=np.array([5,0,0]), 
        number_of_lines=50, 
        color=RED,
        amp_scaler=ValueTracker(1),
        k_scaler=1,
        freq_scaler=1,
        phase_offset=ValueTracker(0.0),
        x_start=None,
        x_end=None,
        y_start=None,
        y_end=None,
        stroke_width=1,
        rotate_angle=None,
        rotate_point=None
    ):
        """Makes an electric field VGroup of Lines.
        Electric field is defined by amp, k, freq, time, and phase ValueTrackers.
        start and end are 3D vectors defining the beginning and end of the wave.
        number_of_lines defines how many Line segments the wave is broken up into.
        x_start and x_end should either be None if we don't want to move the sine wave
        starting points, or a ValueTracker if we do.
        Returns the VGroup.
        """
        # Define the VGroup of Lines which make up the E-field, and store some ValueTrackers
        beam = VGroup()
        beam.amp = amp
        beam.k = k
        beam.freq = freq
        beam.time = time
        beam.phase = phase

        beam.x_start = x_start
        beam.x_end = x_end
        beam.y_start = y_start
        beam.y_end = y_end

        beam.amp_scaler = amp_scaler
        beam.k_scaler = k_scaler
        beam.freq_scaler = freq_scaler
        beam.phase_offset = phase_offset

        amp_float = amp_scaler.get_value() * amp.get_value()
        k_float = k_scaler * k.get_value()
        freq_float = freq_scaler * freq.get_value()
        phase_float = phase_offset.get_value() + phase.get_value()

        # Check if ValueTrackers exist for start and end pos
        if x_start:
            start[0] = x_start.get_value()
        if x_end:
            end[0] = x_end.get_value()

        if y_start:
            start[1] = y_start.get_value()
        if y_end:
            end[1] = y_end.get_value()

        # Get the start and end vector info.
        # This function will first calculate the wave along the x axis,
        # then rotate it appropriately.
        span = end - start
        origin = (start + end) / 2
        total_length = np.sqrt(np.sum(span**2))
        norm = span / total_length
        theta = np.arctan2(norm[1], norm[0])

        x_diff = total_length / number_of_lines

        phase_output = self.get_phase(k=k_float, x=total_length, freq=freq_float, time=time, phase=0)

        # Store some additional info
        beam.phase_output = phase_output
        beam.start = start
        beam.end = end
        beam.origin = origin
        beam.theta = theta
        beam.number_of_lines = number_of_lines
        beam.color = color
        beam.stroke_width = stroke_width
        beam.rotate_angle = rotate_angle
        beam.rotate_point = rotate_point

        for ii in range(number_of_lines):
            # Calculate wave in 1D
            temp_x_start = ii * x_diff
            temp_x_end = (ii + 1) * x_diff

            temp_y_start = self.get_sine_wave(amp_float, k_float, temp_x_start, freq_float, time, phase_float)
            temp_y_end = self.get_sine_wave(amp_float, k_float, temp_x_end, freq_float, time, phase_float)

            # Translate to origin = (0, 0)
            temp_x_start -= total_length/2
            temp_x_end -= total_length/2
            # temp_y_start += origin[1]
            # temp_y_end += origin[1]

            # Rotate about origin
            temp_x_start, temp_y_start = self.rotate(temp_x_start, temp_y_start, theta)
            temp_x_end, temp_y_end = self.rotate(temp_x_end, temp_y_end, theta)

            # Translate to sine wave position beam.origin
            temp_x_start += origin[0]
            temp_x_end += origin[0]
            temp_y_start += origin[1]
            temp_y_end += origin[1]

            # Define the Line
            temp_start = np.array([temp_x_start, temp_y_start, 0])
            temp_end = np.array([temp_x_end, temp_y_end, 0])

            temp_line = Line(temp_start, temp_end, color=color, stroke_width=stroke_width)
            beam.add(temp_line)

        if rotate_angle:
            if not rotate_point:
                rotate_point = np.array([(start[0] + end[0])/2.0, (start[1] + end[1])/2.0, 0.0])
            beam.rotate(rotate_angle, about_point=rotate_point)

        return beam

    def sum_electric_fields(
        self, 
        field1, 
        field2, 
        color=RED, 
        reverse=False, 
        y_axis=None,
        stroke_width=None,
        sign_flip=False,
    ):
        """Add together two VGroups of electric fields.
        y_axis should be a ValueTracker defining the height of the summed wave.
        sign_flip flips the sign of the second beam being summed, useful for phase flip problems
        """
        if field1.number_of_lines != field2.number_of_lines:
            print()
            print("Cannot sum two fields of different lengths")
            return
        
        beam = VGroup()
        beam.field1 = field1
        beam.field2 = field2
        beam.color = color
        beam.reverse = reverse
        beam.y_axis = y_axis
        beam.stroke_width = stroke_width
        beam.sign_flip = sign_flip

        if field1.y_start:
            nominal_y1 = (field1.y_start.get_value() + field1.y_end.get_value())/2.0
        else:
            nominal_y1 = 0

        if field2.y_start:
            nominal_y2 = (field2.y_start.get_value() + field2.y_end.get_value())/2.0
        else:
            nominal_y2 = 0

        if y_axis:
            y_axis_value = y_axis.get_value()
        else:
            y_axis_value = (field1.y_start.get_value() + field1.y_end.get_value())/2.0

        if not stroke_width:
            stroke_width = 4
            beam.stroke_width = 4

        if beam.sign_flip:
            sign = -1
        else:
            sign = 1
        

        for ii in range(field1.number_of_lines):
            if reverse:
                jj = (field1.number_of_lines - 1) - ii
            else:
                jj = ii
        
            line1 = field1[ii]
            line2 = field2[jj]

            if reverse:
                new_line_start = line1.start
                new_line_end = line1.end
                new_line_start[1] = (line1.start[1] - nominal_y1) + sign * (line2.end[1] - nominal_y2) + y_axis_value
                new_line_end[1] = (line1.end[1] - nominal_y1) + sign * (line2.start[1] - nominal_y2) + y_axis_value
            else:
                new_line_start = line1.start
                new_line_end = line1.end
                new_line_start[1] = (line1.start[1] - nominal_y1) + sign * (line2.start[1] - nominal_y2) + y_axis_value
                new_line_end[1] = (line1.end[1] - nominal_y1) + sign * (line2.end[1] - nominal_y2) + y_axis_value

            new_line = Line(new_line_start, new_line_end, color=color, stroke_width=stroke_width)
            beam.add(new_line)

        # rotate
        if field1.rotate_angle:
            if not field1.rotate_point:
                rotate_point = np.array(
                    [(field1.x_start.get_value() + field1.x_end.get_value())/2.0, (field1.y_start.get_value() + field1.y_end.get_value())/2.0, 0.0]
                )
            else:
                rotate_point = field1.rotate_point
            beam.rotate(field1.rotate_angle)

        return beam

    def construct(self):

        self.timer = 0
        self.rate = 0.5

        stroke_width = 2
        number_of_lines = 400

        main_beam_y = -2.0

        ## Electric fields
        # Input Beam 
        amp_input = ValueTracker(0.05)
        k_input = ValueTracker((PI / 2) * 8)
        freq_input = ValueTracker(1)
        time_input = 0.0
        phase_input = ValueTracker(0.0)

        x_start_input = ValueTracker(-7.0)
        x_end_input = ValueTracker(-5.5)
        y_start_input = ValueTracker(main_beam_y)
        y_end_input = ValueTracker(main_beam_y)
        
        start_input = np.array([x_start_input.get_value(), y_start_input.get_value(), 0.0])
        end_input = np.array([x_end_input.get_value(), y_end_input.get_value(), 0.0])

        rotate_angle_input = 0

        self.input_beam = self.make_electric_field(
            amp_input, 
            k_input, 
            freq_input, 
            time_input, 
            phase_input, 
            start_input, 
            end_input, 
            number_of_lines,
            color=RED,
            x_start=x_start_input,
            x_end=x_end_input,
            y_start=y_start_input,
            y_end=y_end_input,
            stroke_width=stroke_width,
            rotate_angle=rotate_angle_input
        )

        # IMC beam 1
        amp_imc_1 = amp_input
        k_imc_1 = k_input
        freq_imc_1 = freq_input
        time_imc_1 = 0.0
        phase_imc_1 = phase_input

        x_start_imc_1 = x_end_input
        x_end_imc_1 = ValueTracker(-3.5)
        y_start_imc_1 = ValueTracker(main_beam_y)
        y_end_imc_1 = ValueTracker(main_beam_y)
        
        start_imc_1 = np.array([x_start_imc_1.get_value(), y_start_imc_1.get_value(), 0.0])
        end_imc_1 = np.array([x_end_imc_1.get_value(), y_end_imc_1.get_value(), 0.0])

        rotate_angle_imc_1 = 0

        amp_imc_1_scaler = ValueTracker(2.0)
        k_imc_1_scaler = 1
        freq_imc_1_scaler = 1
        phase_imc_1_offset = ValueTracker(0.0)
        phase_imc_1_offset.add_updater(
            lambda mob: mob.set_value( 
                self.get_phase(
                    k_input.get_value(), 
                    x_end_input.get_value() - x_start_input.get_value(), 
                    freq_input.get_value(), 
                    time=0, # this is already account for in wave_updater
                    phase=0 # this is already accounted for in phase_refl
                ) 
            )
        )

        self.imc_1_beam = self.make_electric_field(
            amp_imc_1, 
            k_imc_1, 
            freq_imc_1, 
            time_imc_1, 
            phase_imc_1, 
            start_imc_1, 
            end_imc_1, 
            number_of_lines,
            color=RED,
            amp_scaler=amp_imc_1_scaler,
            k_scaler=k_imc_1_scaler,
            freq_scaler=freq_imc_1_scaler,
            phase_offset=phase_imc_1_offset,
            x_start=x_start_imc_1,
            x_end=x_end_imc_1,
            y_start=y_start_imc_1,
            y_end=y_end_imc_1,
            stroke_width=stroke_width,
            rotate_angle=rotate_angle_imc_1
        )

        # IMC beam 2
        amp_imc_2 = amp_input
        k_imc_2 = k_input
        freq_imc_2 = freq_input
        time_imc_2 = 0.0
        phase_imc_2 = phase_input

        # unrotates allows you to put in the start and end points you want for the E-field
        # then manipulates so you can just input the (x,y) coords and theta to make_electric_field and it *just works*
        x1, x2, y1, y2, theta = unrotate(x_end_imc_1.get_value(), x_end_imc_1.get_value()-1, main_beam_y, main_beam_y + 2.5)

        x_start_imc_2 = ValueTracker(x1)
        x_end_imc_2 = ValueTracker(x2)
        y_start_imc_2 = ValueTracker(y1)
        y_end_imc_2 = ValueTracker(y2)
        
        start_imc_2 = np.array([x_start_imc_2.get_value(), y_start_imc_2.get_value(), 0.0])
        end_imc_2 = np.array([x_end_imc_2.get_value(), y_end_imc_2.get_value(), 0.0])

        rotate_angle_imc_2 = theta

        amp_imc_2_scaler = ValueTracker(2.0)
        k_imc_2_scaler = 1
        freq_imc_2_scaler = 1
        phase_imc_2_offset = ValueTracker(0.0)
        phase_imc_2_offset.add_updater(
            lambda mob: mob.set_value( 
                self.get_phase(
                    k_input.get_value(), 
                    x_end_input.get_value() - x_start_input.get_value(), 
                    freq_input.get_value(), 
                    time=0, # this is already account for in wave_updater
                    phase=0 # this is already accounted for in phase_refl
                ) 
            )
        )

        self.imc_2_beam = self.make_electric_field(
            amp_imc_2, 
            k_imc_2, 
            freq_imc_2, 
            time_imc_2, 
            phase_imc_2, 
            start_imc_2, 
            end_imc_2, 
            number_of_lines,
            color=RED,
            amp_scaler=amp_imc_2_scaler,
            k_scaler=k_imc_2_scaler,
            freq_scaler=freq_imc_2_scaler,
            phase_offset=phase_imc_2_offset,
            x_start=x_start_imc_2,
            x_end=x_end_imc_2,
            y_start=y_start_imc_2,
            y_end=y_end_imc_2,
            stroke_width=stroke_width,
            rotate_angle=rotate_angle_imc_2
        )

        # IMC beam 3
        amp_imc_3 = amp_input
        k_imc_3 = k_input
        freq_imc_3 = freq_input
        time_imc_3 = 0.0
        phase_imc_3 = phase_input

        # unrotates allows you to put in the start and end points you want for the E-field
        # then manipulates so you can just input the (x,y) coords and theta to make_electric_field and it *just works*
        x1, x2, y1, y2, theta = unrotate(x_end_imc_1.get_value()-1, x_end_imc_1.get_value()-2, main_beam_y+2.5, main_beam_y)
        
        x_start_imc_3 = ValueTracker(x1)
        x_end_imc_3 = ValueTracker(x2)
        y_start_imc_3 = ValueTracker(y1)
        y_end_imc_3 = ValueTracker(y2)
        
        start_imc_3 = np.array([x_start_imc_3.get_value(), y_start_imc_3.get_value(), 0.0])
        end_imc_3 = np.array([x_end_imc_3.get_value(), y_end_imc_3.get_value(), 0.0])

        rotate_angle_imc_3 = theta

        amp_imc_3_scaler = ValueTracker(2.0)
        k_imc_3_scaler = 1
        freq_imc_3_scaler = 1
        phase_imc_3_offset = ValueTracker(0.0)
        phase_imc_3_offset.add_updater(
            lambda mob: mob.set_value( 
                self.get_phase(
                    k_input.get_value(), 
                    x_end_input.get_value() - x_start_input.get_value(), 
                    freq_input.get_value(), 
                    time=0, # this is already account for in wave_updater
                    phase=0 # this is already accounted for in phase_refl
                ) 
            )
        )

        self.imc_3_beam = self.make_electric_field(
            amp_imc_3, 
            k_imc_3, 
            freq_imc_3, 
            time_imc_3, 
            phase_imc_3, 
            start_imc_3, 
            end_imc_3, 
            number_of_lines,
            color=RED,
            amp_scaler=amp_imc_3_scaler,
            k_scaler=k_imc_3_scaler,
            freq_scaler=freq_imc_3_scaler,
            phase_offset=phase_imc_3_offset,
            x_start=x_start_imc_3,
            x_end=x_end_imc_3,
            y_start=y_start_imc_3,
            y_end=y_end_imc_3,
            stroke_width=stroke_width,
            rotate_angle=rotate_angle_imc_3
        )

        # post IMC beam
        amp_post_imc = amp_input
        k_post_imc = k_input
        freq_post_imc = freq_input
        time_post_imc = 0.0
        phase_post_imc = phase_input

        x_start_post_imc = x_end_imc_1
        x_end_post_imc = ValueTracker(-2.0)
        y_start_post_imc = ValueTracker(main_beam_y)
        y_end_post_imc = ValueTracker(main_beam_y)
        
        start_post_imc = np.array([x_start_post_imc.get_value(), y_start_post_imc.get_value(), 0.0])
        end_post_imc = np.array([x_end_post_imc.get_value(), y_end_post_imc.get_value(), 0.0])

        rotate_angle_post_imc = 0

        amp_post_imc_scaler = ValueTracker(1.0)
        k_post_imc_scaler = 1
        freq_post_imc_scaler = 1
        phase_post_imc_offset = ValueTracker(0.0)
        phase_post_imc_offset.add_updater(
            lambda mob: mob.set_value( 
                self.get_phase(
                    k_input.get_value(), 
                    x_end_input.get_value() - x_end_imc_1.get_value(), 
                    freq_input.get_value(), 
                    time=0, # this is already account for in wave_updater
                    phase=0 # this is already accounted for in phase_refl
                ) 
            )
        )

        self.post_imc_beam = self.make_electric_field(
            amp_post_imc, 
            k_post_imc, 
            freq_post_imc, 
            time_post_imc, 
            phase_post_imc, 
            start_post_imc, 
            end_post_imc, 
            number_of_lines,
            color=RED,
            amp_scaler=amp_post_imc_scaler,
            k_scaler=k_post_imc_scaler,
            freq_scaler=freq_post_imc_scaler,
            phase_offset=phase_post_imc_offset,
            x_start=x_start_post_imc,
            x_end=x_end_post_imc,
            y_start=y_start_post_imc,
            y_end=y_end_post_imc,
            stroke_width=stroke_width,
            rotate_angle=rotate_angle_post_imc
        )

        # PRC to BS forward beam
        amp_prc_bs_1 = amp_input
        k_prc_bs_1 = k_input
        freq_prc_bs_1 = freq_input
        time_prc_bs_1 = 0.0
        phase_prc_bs_1 = phase_input

        x_start_prc_bs_1 = x_end_post_imc
        x_end_prc_bs_1 = ValueTracker(0.0)
        y_start_prc_bs_1 = ValueTracker(main_beam_y)
        y_end_prc_bs_1 = ValueTracker(main_beam_y)
        
        start_prc_bs_1 = np.array([x_start_prc_bs_1.get_value(), y_start_prc_bs_1.get_value(), 0.0])
        end_prc_bs_1 = np.array([x_end_prc_bs_1.get_value(), y_end_prc_bs_1.get_value(), 0.0])

        rotate_angle_prc_bs_1 = 0

        amp_prc_bs_1_scaler = ValueTracker(2.0)
        k_prc_bs_1_scaler = 1
        freq_prc_bs_1_scaler = 1
        phase_prc_bs_1_offset = ValueTracker(0.0)
        phase_prc_bs_1_offset.add_updater(
            lambda mob: mob.set_value( 
                self.get_phase(
                    k_input.get_value(), 
                    x_end_post_imc.get_value() - x_start_input.get_value(), 
                    freq_input.get_value(), 
                    time=0, # this is already account for in wave_updater
                    phase=0 # this is already accounted for in phase_refl
                ) 
            )
        )

        self.prc_bs_1_beam = self.make_electric_field(
            amp_prc_bs_1, 
            k_prc_bs_1, 
            freq_prc_bs_1, 
            time_prc_bs_1, 
            phase_prc_bs_1, 
            start_prc_bs_1, 
            end_prc_bs_1, 
            number_of_lines,
            color=RED,
            amp_scaler=amp_prc_bs_1_scaler,
            k_scaler=k_prc_bs_1_scaler,
            freq_scaler=freq_prc_bs_1_scaler,
            phase_offset=phase_prc_bs_1_offset,
            x_start=x_start_prc_bs_1,
            x_end=x_end_prc_bs_1,
            y_start=y_start_prc_bs_1,
            y_end=y_end_prc_bs_1,
            stroke_width=stroke_width,
            rotate_angle=rotate_angle_prc_bs_1
        )

        # PRC to BS backward beam
        amp_prc_bs_2 = amp_input
        k_prc_bs_2 = k_input
        freq_prc_bs_2 = freq_input
        time_prc_bs_2 = 0.0
        phase_prc_bs_2 = phase_input

        x_start_prc_bs_2 = x_start_prc_bs_1 
        x_end_prc_bs_2 = x_end_prc_bs_1
        y_start_prc_bs_2 = y_start_prc_bs_1
        y_end_prc_bs_2 = y_end_prc_bs_1
        
        start_prc_bs_2 = np.array([x_start_prc_bs_2.get_value(), y_start_prc_bs_2.get_value(), 0.0])
        end_prc_bs_2 = np.array([x_end_prc_bs_2.get_value(), y_end_prc_bs_2.get_value(), 0.0])

        rotate_angle_prc_bs_2 = PI

        amp_prc_bs_2_scaler = ValueTracker(2.0)
        k_prc_bs_2_scaler = -1
        freq_prc_bs_2_scaler = 1
        phase_prc_bs_2_offset = ValueTracker(0.0)
        phase_prc_bs_2_offset.add_updater(
            lambda mob: mob.set_value( 
                self.get_phase(
                    k_input.get_value(), 
                    x_end_post_imc.get_value() - x_start_input.get_value(), 
                    freq_input.get_value(), 
                    time=0, # this is already account for in wave_updater
                    phase=0 # this is already accounted for in phase_refl
                ) 
            )
        )

        self.prc_bs_2_beam = self.make_electric_field(
            amp_prc_bs_2, 
            k_prc_bs_2, 
            freq_prc_bs_2, 
            time_prc_bs_2, 
            phase_prc_bs_2, 
            start_prc_bs_2, 
            end_prc_bs_2, 
            number_of_lines,
            color=RED,
            amp_scaler=amp_prc_bs_2_scaler,
            k_scaler=k_prc_bs_2_scaler,
            freq_scaler=freq_prc_bs_2_scaler,
            phase_offset=phase_prc_bs_2_offset,
            x_start=x_start_prc_bs_2,
            x_end=x_end_prc_bs_2,
            y_start=y_start_prc_bs_2,
            y_end=y_end_prc_bs_2,
            stroke_width=stroke_width,
            rotate_angle=rotate_angle_prc_bs_2
        )

        # PRC BS standing wave
        self.prc_bs_beam = self.sum_electric_fields(self.prc_bs_1_beam, self.prc_bs_2_beam, color=RED, reverse=True, stroke_width=4)

        # BS to Y-arm forward beam
        amp_bs_y_1 = amp_input
        k_bs_y_1 = k_input
        freq_bs_y_1 = freq_input
        time_bs_y_1 = 0.0
        phase_bs_y_1 = phase_input

        x1, x2, y1, y2, theta = unrotate(
            x_end_prc_bs_1.get_value(), 
            x_end_prc_bs_1.get_value(), 
            y_start_prc_bs_1.get_value(), 
            y_start_prc_bs_1.get_value() + 1.0
        )
        x_start_bs_y_1 = ValueTracker(x1)
        x_end_bs_y_1 = ValueTracker(x2)
        y_start_bs_y_1 = ValueTracker(y1)
        y_end_bs_y_1 = ValueTracker(y2)
        
        start_bs_y_1 = np.array([x_start_bs_y_1.get_value(), y_start_bs_y_1.get_value(), 0.0])
        end_bs_y_1 = np.array([x_end_bs_y_1.get_value(), y_end_bs_y_1.get_value(), 0.0])

        rotate_angle_bs_y_1 = theta

        amp_bs_y_1_scaler = ValueTracker(2.0/np.sqrt(2.0))
        k_bs_y_1_scaler = 1
        freq_bs_y_1_scaler = 1
        phase_bs_y_1_offset = ValueTracker(0.0)
        phase_bs_y_1_offset.add_updater(
            lambda mob: mob.set_value( 
                self.get_phase(
                    k_input.get_value(), 
                    x_end_prc_bs_1.get_value() - x_start_input.get_value(), 
                    freq_input.get_value(), 
                    time=0, # this is already account for in wave_updater
                    phase=0 # this is already accounted for in phase_refl
                ) 
            )
        )

        self.bs_y_1_beam = self.make_electric_field(
            amp_bs_y_1, 
            k_bs_y_1, 
            freq_bs_y_1, 
            time_bs_y_1, 
            phase_bs_y_1, 
            start_bs_y_1, 
            end_bs_y_1, 
            number_of_lines,
            color=RED,
            amp_scaler=amp_bs_y_1_scaler,
            k_scaler=k_bs_y_1_scaler,
            freq_scaler=freq_bs_y_1_scaler,
            phase_offset=phase_bs_y_1_offset,
            x_start=x_start_bs_y_1,
            x_end=x_end_bs_y_1,
            y_start=y_start_bs_y_1,
            y_end=y_end_bs_y_1,
            stroke_width=stroke_width,
            rotate_angle=rotate_angle_bs_y_1
        )

        # BS to Y-arm backward beam
        amp_bs_y_2 = amp_input
        k_bs_y_2 = k_input
        freq_bs_y_2 = freq_input
        time_bs_y_2 = 0.0
        phase_bs_y_2 = phase_input

        x1, x2, y1, y2, theta = unrotate(
            x_end_prc_bs_1.get_value(), 
            x_end_prc_bs_1.get_value(), 
            y_start_prc_bs_1.get_value() + 1.0, 
            y_start_prc_bs_1.get_value()
        )
        x_start_bs_y_2 = ValueTracker(x1)
        x_end_bs_y_2 = ValueTracker(x2)
        y_start_bs_y_2 = ValueTracker(y1)
        y_end_bs_y_2 = ValueTracker(y2)
        
        start_bs_y_2 = np.array([x_start_bs_y_2.get_value(), y_start_bs_y_2.get_value(), 0.0])
        end_bs_y_2 = np.array([x_end_bs_y_2.get_value(), y_end_bs_y_2.get_value(), 0.0])

        rotate_angle_bs_y_2 = theta

        amp_bs_y_2_scaler = ValueTracker(2.0/np.sqrt(2.0))
        k_bs_y_2_scaler = 1
        freq_bs_y_2_scaler = 1
        phase_bs_y_2_offset = ValueTracker(PI)
        phase_bs_y_2_offset.add_updater(
            lambda mob: mob.set_value( 
                self.get_phase(
                    k_input.get_value(), 
                    x_end_prc_bs_1.get_value() - x_start_input.get_value(), 
                    freq_input.get_value(), 
                    time=0, # this is already account for in wave_updater
                    phase=0 # this is already accounted for in phase_refl
                ) 
            )
        )

        self.bs_y_2_beam = self.make_electric_field(
            amp_bs_y_2, 
            k_bs_y_2, 
            freq_bs_y_2, 
            time_bs_y_2, 
            phase_bs_y_2, 
            start_bs_y_2, 
            end_bs_y_2, 
            number_of_lines,
            color=RED,
            amp_scaler=amp_bs_y_2_scaler,
            k_scaler=k_bs_y_2_scaler,
            freq_scaler=freq_bs_y_2_scaler,
            phase_offset=phase_bs_y_2_offset,
            x_start=x_start_bs_y_2,
            x_end=x_end_bs_y_2,
            y_start=y_start_bs_y_2,
            y_end=y_end_bs_y_2,
            stroke_width=stroke_width,
            rotate_angle=rotate_angle_bs_y_2
        )

        # BS to Y-arm standing wave
        self.bs_y_beam = self.sum_electric_fields(self.bs_y_1_beam, self.bs_y_2_beam, color=RED, reverse=True, sign_flip=True, stroke_width=3)

        # BS to X-arm forward beam
        amp_bs_x_1 = amp_input
        k_bs_x_1 = k_input
        freq_bs_x_1 = freq_input
        time_bs_x_1 = 0.0
        phase_bs_x_1 = phase_input

        x_start_bs_x_1 = x_end_prc_bs_1
        x_end_bs_x_1 = ValueTracker(1.0)
        y_start_bs_x_1 = ValueTracker(main_beam_y)
        y_end_bs_x_1 = ValueTracker(main_beam_y)
        
        start_bs_x_1 = np.array([x_start_bs_x_1.get_value(), y_start_bs_x_1.get_value(), 0.0])
        end_bs_x_1 = np.array([x_end_bs_x_1.get_value(), y_end_bs_x_1.get_value(), 0.0])

        rotate_angle_bs_x_1 = 0

        amp_bs_x_1_scaler = ValueTracker(2.0/np.sqrt(2.0))
        k_bs_x_1_scaler = 1
        freq_bs_x_1_scaler = 1
        phase_bs_x_1_offset = ValueTracker(0.0)
        phase_bs_x_1_offset.add_updater(
            lambda mob: mob.set_value( 
                self.get_phase(
                    k_input.get_value(), 
                    x_end_prc_bs_1.get_value() - x_start_input.get_value(), 
                    freq_input.get_value(), 
                    time=0, # this is already account for in wave_updater
                    phase=0 # this is already accounted for in phase_refl
                ) 
            )
        )

        self.bs_x_1_beam = self.make_electric_field(
            amp_bs_x_1, 
            k_bs_x_1, 
            freq_bs_x_1, 
            time_bs_x_1, 
            phase_bs_x_1, 
            start_bs_x_1, 
            end_bs_x_1, 
            number_of_lines,
            color=RED,
            amp_scaler=amp_bs_x_1_scaler,
            k_scaler=k_bs_x_1_scaler,
            freq_scaler=freq_bs_x_1_scaler,
            phase_offset=phase_bs_x_1_offset,
            x_start=x_start_bs_x_1,
            x_end=x_end_bs_x_1,
            y_start=y_start_bs_x_1,
            y_end=y_end_bs_x_1,
            stroke_width=stroke_width,
            rotate_angle=rotate_angle_bs_x_1
        )

        # BS to X-arm backward beam
        amp_bs_x_2 = amp_input
        k_bs_x_2 = k_input
        freq_bs_x_2 = freq_input
        time_bs_x_2 = 0.0
        phase_bs_x_2 = phase_input

        x_start_bs_x_2 = x_end_prc_bs_1
        x_end_bs_x_2 = ValueTracker(1.0)
        y_start_bs_x_2 = ValueTracker(main_beam_y)
        y_end_bs_x_2 = ValueTracker(main_beam_y)
        
        start_bs_x_2 = np.array([x_start_bs_x_2.get_value(), y_start_bs_x_2.get_value(), 0.0])
        end_bs_x_2 = np.array([x_end_bs_x_2.get_value(), y_end_bs_x_2.get_value(), 0.0])

        rotate_angle_bs_x_2 = PI

        amp_bs_x_2_scaler = ValueTracker(2.0/np.sqrt(2.0))
        k_bs_x_2_scaler = -1
        freq_bs_x_2_scaler = 1
        phase_bs_x_2_offset = ValueTracker(0.0)
        phase_bs_x_2_offset.add_updater(
            lambda mob: mob.set_value( 
                self.get_phase(
                    k_input.get_value(), 
                    x_end_prc_bs_1.get_value() - x_start_input.get_value(), 
                    freq_input.get_value(), 
                    time=0, # this is already account for in wave_updater
                    phase=0 # this is already accounted for in phase_refl
                ) 
            )
        )

        self.bs_x_2_beam = self.make_electric_field(
            amp_bs_x_2, 
            k_bs_x_2, 
            freq_bs_x_2, 
            time_bs_x_2, 
            phase_bs_x_2, 
            start_bs_x_2, 
            end_bs_x_2, 
            number_of_lines,
            color=RED,
            amp_scaler=amp_bs_x_2_scaler,
            k_scaler=k_bs_x_2_scaler,
            freq_scaler=freq_bs_x_2_scaler,
            phase_offset=phase_bs_x_2_offset,
            x_start=x_start_bs_x_2,
            x_end=x_end_bs_x_2,
            y_start=y_start_bs_x_2,
            y_end=y_end_bs_x_2,
            stroke_width=stroke_width,
            rotate_angle=rotate_angle_bs_x_2
        )

        # BS to X-arm standing wave
        self.bs_x_beam = self.sum_electric_fields(self.bs_x_1_beam, self.bs_x_2_beam, color=RED, reverse=True, stroke_width=3)

        # Y-arm forward beam
        amp_y_arm_1 = amp_input
        k_y_arm_1 = k_input
        freq_y_arm_1 = freq_input
        time_y_arm_1 = 0.0
        phase_y_arm_1 = phase_input

        x1, x2, y1, y2, theta = unrotate(
            x_end_prc_bs_1.get_value(), 
            x_end_prc_bs_1.get_value(), 
            y_start_prc_bs_1.get_value() + 1.0, 
            3.0
        )
        x_start_y_arm_1 = ValueTracker(x1)
        x_end_y_arm_1 = ValueTracker(x2)
        y_start_y_arm_1 = ValueTracker(y1)
        y_end_y_arm_1 = ValueTracker(y2)
        
        start_y_arm_1 = np.array([x_start_y_arm_1.get_value(), y_start_y_arm_1.get_value(), 0.0])
        end_y_arm_1 = np.array([x_end_y_arm_1.get_value(), y_end_y_arm_1.get_value(), 0.0])

        rotate_angle_y_arm_1 = theta

        amp_y_arm_1_scaler = ValueTracker(5.0)
        k_y_arm_1_scaler = 1
        freq_y_arm_1_scaler = 1
        phase_y_arm_1_offset = ValueTracker(0.0)
        phase_y_arm_1_offset.add_updater(
            lambda mob: mob.set_value( 
                self.get_phase(
                    k_input.get_value(), 
                    x_end_bs_x_1.get_value() - x_start_input.get_value(), 
                    freq_input.get_value(), 
                    time=0, # this is already account for in wave_updater
                    phase=0 # this is already accounted for in phase_refl
                ) 
            )
        )

        self.y_arm_1_beam = self.make_electric_field(
            amp_y_arm_1, 
            k_y_arm_1, 
            freq_y_arm_1, 
            time_y_arm_1, 
            phase_y_arm_1, 
            start_y_arm_1, 
            end_y_arm_1, 
            number_of_lines,
            color=RED,
            amp_scaler=amp_y_arm_1_scaler,
            k_scaler=k_y_arm_1_scaler,
            freq_scaler=freq_y_arm_1_scaler,
            phase_offset=phase_y_arm_1_offset,
            x_start=x_start_y_arm_1,
            x_end=x_end_y_arm_1,
            y_start=y_start_y_arm_1,
            y_end=y_end_y_arm_1,
            stroke_width=stroke_width,
            rotate_angle=rotate_angle_y_arm_1
        )

        # Y-arm backward beam
        amp_y_arm_2 = amp_input
        k_y_arm_2 = k_input
        freq_y_arm_2 = freq_input
        time_y_arm_2 = 0.0
        phase_y_arm_2 = phase_input

        x1, x2, y1, y2, theta = unrotate(
            x_end_prc_bs_1.get_value(), 
            x_end_prc_bs_1.get_value(), 
            3.0,
            y_start_prc_bs_1.get_value() + 1.0
        )
        x_start_y_arm_2 = ValueTracker(x1)
        x_end_y_arm_2 = ValueTracker(x2)
        y_start_y_arm_2 = ValueTracker(y1)
        y_end_y_arm_2 = ValueTracker(y2)
        
        start_y_arm_2 = np.array([x_start_y_arm_2.get_value(), y_start_y_arm_2.get_value(), 0.0])
        end_y_arm_2 = np.array([x_end_y_arm_2.get_value(), y_end_y_arm_2.get_value(), 0.0])

        rotate_angle_y_arm_2 = theta

        amp_y_arm_2_scaler = ValueTracker(5.0)
        k_y_arm_2_scaler = 1
        freq_y_arm_2_scaler = 1
        phase_y_arm_2_offset = ValueTracker(PI)
        phase_y_arm_2_offset.add_updater(
            lambda mob: mob.set_value( 
                self.get_phase(
                    k_input.get_value(), 
                    x_end_bs_x_1.get_value() - x_start_input.get_value(), 
                    freq_input.get_value(), 
                    time=0, # this is already account for in wave_updater
                    phase=0 # this is already accounted for in phase_refl
                ) 
            )
        )

        self.y_arm_2_beam = self.make_electric_field(
            amp_y_arm_2, 
            k_y_arm_2, 
            freq_y_arm_2, 
            time_y_arm_2, 
            phase_y_arm_2, 
            start_y_arm_2, 
            end_y_arm_2, 
            number_of_lines,
            color=RED,
            amp_scaler=amp_y_arm_2_scaler,
            k_scaler=k_y_arm_2_scaler,
            freq_scaler=freq_y_arm_2_scaler,
            phase_offset=phase_y_arm_2_offset,
            x_start=x_start_y_arm_2,
            x_end=x_end_y_arm_2,
            y_start=y_start_y_arm_2,
            y_end=y_end_y_arm_2,
            stroke_width=stroke_width,
            rotate_angle=rotate_angle_y_arm_2
        )

        # Y-arm standing wave
        self.y_arm_beam = self.sum_electric_fields(self.y_arm_1_beam, self.y_arm_2_beam, color=RED, reverse=True, sign_flip=True, stroke_width=6)

        # X-arm forward beam
        amp_x_arm_1 = amp_input
        k_x_arm_1 = k_input
        freq_x_arm_1 = freq_input
        time_x_arm_1 = 0.0
        phase_x_arm_1 = phase_input

        x_start_x_arm_1 = x_end_bs_x_1
        x_end_x_arm_1 = ValueTracker(6.0)
        y_start_x_arm_1 = ValueTracker(main_beam_y)
        y_end_x_arm_1 = ValueTracker(main_beam_y)
        
        start_x_arm_1 = np.array([x_start_x_arm_1.get_value(), y_start_x_arm_1.get_value(), 0.0])
        end_x_arm_1 = np.array([x_end_x_arm_1.get_value(), y_end_x_arm_1.get_value(), 0.0])

        rotate_angle_x_arm_1 = 0

        amp_x_arm_1_scaler = ValueTracker(5.0)
        k_x_arm_1_scaler = 1
        freq_x_arm_1_scaler = 1
        phase_x_arm_1_offset = ValueTracker(0.0)
        phase_x_arm_1_offset.add_updater(
            lambda mob: mob.set_value( 
                self.get_phase(
                    k_input.get_value(), 
                    x_end_bs_x_1.get_value() - x_start_input.get_value(), 
                    freq_input.get_value(), 
                    time=0, # this is already account for in wave_updater
                    phase=0 # this is already accounted for in phase_refl
                ) 
            )
        )

        self.x_arm_1_beam = self.make_electric_field(
            amp_x_arm_1, 
            k_x_arm_1, 
            freq_x_arm_1, 
            time_x_arm_1, 
            phase_x_arm_1, 
            start_x_arm_1, 
            end_x_arm_1, 
            number_of_lines,
            color=RED,
            amp_scaler=amp_x_arm_1_scaler,
            k_scaler=k_x_arm_1_scaler,
            freq_scaler=freq_x_arm_1_scaler,
            phase_offset=phase_x_arm_1_offset,
            x_start=x_start_x_arm_1,
            x_end=x_end_x_arm_1,
            y_start=y_start_x_arm_1,
            y_end=y_end_x_arm_1,
            stroke_width=stroke_width,
            rotate_angle=rotate_angle_x_arm_1
        )

        # X-arm backward beam
        amp_x_arm_2 = amp_input
        k_x_arm_2 = k_input
        freq_x_arm_2 = freq_input
        time_x_arm_2 = 0.0
        phase_x_arm_2 = phase_input

        x_start_x_arm_2 = x_end_bs_x_1
        x_end_x_arm_2 = ValueTracker(6.0)
        y_start_x_arm_2 = ValueTracker(main_beam_y)
        y_end_x_arm_2 = ValueTracker(main_beam_y)
        
        start_x_arm_2 = np.array([x_start_x_arm_2.get_value(), y_start_x_arm_2.get_value(), 0.0])
        end_x_arm_2 = np.array([x_end_x_arm_2.get_value(), y_end_x_arm_2.get_value(), 0.0])

        rotate_angle_x_arm_2 = PI

        amp_x_arm_2_scaler = ValueTracker(5.0)
        k_x_arm_2_scaler = -1
        freq_x_arm_2_scaler = 1
        phase_x_arm_2_offset = ValueTracker(0.0)
        phase_x_arm_2_offset.add_updater(
            lambda mob: mob.set_value( 
                self.get_phase(
                    k_input.get_value(), 
                    x_end_bs_x_1.get_value() - x_start_input.get_value(), 
                    freq_input.get_value(), 
                    time=0, # this is already account for in wave_updater
                    phase=0 # this is already accounted for in phase_refl
                ) 
            )
        )

        self.x_arm_2_beam = self.make_electric_field(
            amp_x_arm_2, 
            k_x_arm_2, 
            freq_x_arm_2, 
            time_x_arm_2, 
            phase_x_arm_2, 
            start_x_arm_2, 
            end_x_arm_2, 
            number_of_lines,
            color=RED,
            amp_scaler=amp_x_arm_2_scaler,
            k_scaler=k_x_arm_2_scaler,
            freq_scaler=freq_x_arm_2_scaler,
            phase_offset=phase_x_arm_2_offset,
            x_start=x_start_x_arm_2,
            x_end=x_end_x_arm_2,
            y_start=y_start_x_arm_2,
            y_end=y_end_x_arm_2,
            stroke_width=stroke_width,
            rotate_angle=rotate_angle_x_arm_2
        )

        # X-arm standing wave
        self.x_arm_beam = self.sum_electric_fields(self.x_arm_1_beam, self.x_arm_2_beam, color=RED, reverse=True, stroke_width=6)

        # BS to SRC beam forward beam
        amp_bs_src_1 = amp_input
        k_bs_src_1 = k_input
        freq_bs_src_1 = freq_input
        time_bs_src_1 = 0.0
        phase_bs_src_1 = phase_input

        x1, x2, y1, y2, theta = unrotate(
            x_end_prc_bs_1.get_value(), 
            x_end_prc_bs_1.get_value(), 
            y_start_prc_bs_1.get_value(), 
            y_start_prc_bs_1.get_value() - 1.0
        )
        x_start_bs_src_1 = ValueTracker(x1)
        x_end_bs_src_1 = ValueTracker(x2)
        y_start_bs_src_1 = ValueTracker(y1)
        y_end_bs_src_1 = ValueTracker(y2)
        
        start_bs_src_1 = np.array([x_start_bs_src_1.get_value(), y_start_bs_src_1.get_value(), 0.0])
        end_bs_src_1 = np.array([x_end_bs_src_1.get_value(), y_end_bs_src_1.get_value(), 0.0])

        rotate_angle_bs_src_1 = theta

        amp_bs_src_1_scaler = ValueTracker(1.0)
        k_bs_src_1_scaler = 1
        freq_bs_src_1_scaler = 1
        phase_bs_src_1_offset = ValueTracker(0.0)
        phase_bs_src_1_offset.add_updater(
            lambda mob: mob.set_value( 
                self.get_phase(
                    k_input.get_value(), 
                    x_end_prc_bs_1.get_value() - x_start_input.get_value(), 
                    freq_input.get_value(), 
                    time=0, # this is already account for in wave_updater
                    phase=0 # this is already accounted for in phase_refl
                ) 
            )
        )

        self.bs_src_1_beam = self.make_electric_field(
            amp_bs_src_1, 
            k_bs_src_1, 
            freq_bs_src_1, 
            time_bs_src_1, 
            phase_bs_src_1, 
            start_bs_src_1, 
            end_bs_src_1, 
            number_of_lines,
            color=RED,
            amp_scaler=amp_bs_src_1_scaler,
            k_scaler=k_bs_src_1_scaler,
            freq_scaler=freq_bs_src_1_scaler,
            phase_offset=phase_bs_src_1_offset,
            x_start=x_start_bs_src_1,
            x_end=x_end_bs_src_1,
            y_start=y_start_bs_src_1,
            y_end=y_end_bs_src_1,
            stroke_width=stroke_width,
            rotate_angle=rotate_angle_bs_src_1
        )

        # BS to SRC beam backward beam
        amp_bs_src_2 = amp_input
        k_bs_src_2 = k_input
        freq_bs_src_2 = freq_input
        time_bs_src_2 = 0.0
        phase_bs_src_2 = phase_input

        x1, x2, y1, y2, theta = unrotate(
            x_end_prc_bs_1.get_value(), 
            x_end_prc_bs_1.get_value(), 
            y_start_prc_bs_1.get_value() - 1.0, 
            y_start_prc_bs_1.get_value()
        )
        x_start_bs_src_2 = ValueTracker(x1)
        x_end_bs_src_2 = ValueTracker(x2)
        y_start_bs_src_2 = ValueTracker(y1)
        y_end_bs_src_2 = ValueTracker(y2)
        
        start_bs_src_2 = np.array([x_start_bs_src_2.get_value(), y_start_bs_src_2.get_value(), 0.0])
        end_bs_src_2 = np.array([x_end_bs_src_2.get_value(), y_end_bs_src_2.get_value(), 0.0])

        rotate_angle_bs_src_2 = theta

        amp_bs_src_2_scaler = ValueTracker(1.0)
        k_bs_src_2_scaler = 1
        freq_bs_src_2_scaler = 1
        phase_bs_src_2_offset = ValueTracker(PI)
        phase_bs_src_2_offset.add_updater(
            lambda mob: mob.set_value( 
                self.get_phase(
                    k_input.get_value(), 
                    x_end_prc_bs_1.get_value() - x_start_input.get_value(), 
                    freq_input.get_value(), 
                    time=0, # this is already account for in wave_updater
                    phase=0 # this is already accounted for in phase_refl
                ) 
            )
        )

        self.bs_src_2_beam = self.make_electric_field(
            amp_bs_src_2, 
            k_bs_src_2, 
            freq_bs_src_2, 
            time_bs_src_2, 
            phase_bs_src_2, 
            start_bs_src_2, 
            end_bs_src_2, 
            number_of_lines,
            color=RED,
            amp_scaler=amp_bs_src_2_scaler,
            k_scaler=k_bs_src_2_scaler,
            freq_scaler=freq_bs_src_2_scaler,
            phase_offset=phase_bs_src_2_offset,
            x_start=x_start_bs_src_2,
            x_end=x_end_bs_src_2,
            y_start=y_start_bs_src_2,
            y_end=y_end_bs_src_2,
            stroke_width=stroke_width,
            rotate_angle=rotate_angle_bs_src_2
        )

        # BS to SRC standing wave
        self.bs_src_beam = self.sum_electric_fields(self.bs_src_1_beam, self.bs_src_2_beam, color=RED, reverse=True, sign_flip=True, stroke_width=2)

        # post SRC beam
        amp_post_src = amp_input
        k_post_src = k_input
        freq_post_src = freq_input
        time_post_src = 0.0
        phase_post_src = phase_input

        x1, x2, y1, y2, theta = unrotate(
            x_end_prc_bs_1.get_value(), 
            x_end_prc_bs_1.get_value(), 
            y_start_prc_bs_1.get_value() - 1.0, 
            y_start_prc_bs_1.get_value() - 1.5
        )
        x_start_post_src = ValueTracker(x1)
        x_end_post_src = ValueTracker(x2)
        y_start_post_src = ValueTracker(y1)
        y_end_post_src = ValueTracker(y2)
        
        start_post_src = np.array([x_start_post_src.get_value(), y_start_post_src.get_value(), 0.0])
        end_post_src = np.array([x_end_post_src.get_value(), y_end_post_src.get_value(), 0.0])

        rotate_angle_post_src = theta

        amp_post_src_scaler = ValueTracker(1.0)
        k_post_src_scaler = 1
        freq_post_src_scaler = 1
        phase_post_src_offset = ValueTracker(0.0)
        phase_post_src_offset.add_updater(
            lambda mob: mob.set_value( 
                self.get_phase(
                    k_input.get_value(), 
                    x_end_prc_bs_1.get_value() - x_start_input.get_value(), 
                    freq_input.get_value(), 
                    time=0, # this is already account for in wave_updater
                    phase=0 # this is already accounted for in phase_refl
                ) 
            )
        )

        self.post_src_beam = self.make_electric_field(
            amp_post_src, 
            k_post_src, 
            freq_post_src, 
            time_post_src, 
            phase_post_src, 
            start_post_src, 
            end_post_src, 
            number_of_lines,
            color=RED,
            amp_scaler=amp_post_src_scaler,
            k_scaler=k_post_src_scaler,
            freq_scaler=freq_post_src_scaler,
            phase_offset=phase_post_src_offset,
            x_start=x_start_post_src,
            x_end=x_end_post_src,
            y_start=y_start_post_src,
            y_end=y_end_post_src,
            stroke_width=stroke_width,
            rotate_angle=rotate_angle_post_src
        )

        ## Photodetector
        pd_center = np.array([x_end_prc_bs_1.get_value(), y_start_prc_bs_1.get_value() - 1.5, 0.0])
        self.photodetector = Sector(start_angle=PI, angle=PI, arc_center=pd_center, outer_radius=0.2, color=GREEN)

        ## Mirrors
        small_mirror_width = 0.1
        small_mirror_height = 0.3

        mid_mirror_width = 0.15
        mid_mirror_height = 0.6

        large_mirror_width = 0.3
        large_mirror_height = 1.0

        # MC1
        mc1_position = np.array([x_start_imc_1.get_value(), main_beam_y, 0.0])
        mc1_rotation = PI/4
        self.mc1 = Rectangle(
            color=BLUE, width=small_mirror_width, height=small_mirror_height, fill_opacity=0.5, stroke_width=2,
        ).move_to(mc1_position - small_mirror_width/2 * RIGHT).rotate(mc1_rotation, about_point=mc1_position)

        # MC2
        mc2_position = np.array([x_end_imc_1.get_value()-1, main_beam_y + 2.5, 0.0])
        mc2_rotation = 3*PI/2
        self.mc2 = Rectangle(
            color=BLUE, width=small_mirror_width, height=small_mirror_height, fill_opacity=0.5, stroke_width=2,
        ).move_to(mc2_position - small_mirror_width/2 * RIGHT).rotate(mc2_rotation, about_point=mc2_position)

        # MC3
        mc3_position = np.array([x_end_imc_1.get_value(), main_beam_y, 0.0])
        mc3_rotation = 3*PI/4
        self.mc3 = Rectangle(
            color=BLUE, width=small_mirror_width, height=small_mirror_height, fill_opacity=0.5, stroke_width=2,
        ).move_to(mc3_position - small_mirror_width/2 * RIGHT).rotate(mc3_rotation, about_point=mc3_position)

        # PRM
        prm_position = np.array([x_start_prc_bs_1.get_value(), main_beam_y, 0.0])
        prm_rotation = 0.0
        self.prm = Rectangle(
            color=BLUE, width=mid_mirror_width, height=mid_mirror_height, fill_opacity=0.5, stroke_width=2,
        ).move_to(prm_position - mid_mirror_width/2 * RIGHT).rotate(prm_rotation, about_point=prm_position)

        # BS
        bs_position = np.array([x_end_prc_bs_1.get_value(), main_beam_y, 0.0])
        bs_rotation = 3*PI/4
        self.bs = Rectangle(
            color=BLUE, width=mid_mirror_width, height=mid_mirror_height, fill_opacity=0.5, stroke_width=2,
        ).move_to(bs_position - mid_mirror_width/2 * RIGHT).rotate(bs_rotation, about_point=bs_position)

        # SRM
        srm_position = np.array([x_end_prc_bs_1.get_value(), main_beam_y - 1.0, 0.0])
        srm_rotation = PI/2
        self.srm = Rectangle(
            color=BLUE, width=mid_mirror_width, height=mid_mirror_height, fill_opacity=0.5, stroke_width=2,
        ).move_to(srm_position - mid_mirror_width/2 * RIGHT).rotate(srm_rotation, about_point=srm_position)

        # ITMX
        itmx_position = np.array([x_end_bs_x_1.get_value(), main_beam_y, 0.0])
        itmx_rotation = 0
        self.itmx = Rectangle(
            color=BLUE, width=large_mirror_width, height=large_mirror_height, fill_opacity=0.5, stroke_width=2,
        ).move_to(itmx_position - large_mirror_width/2 * RIGHT).rotate(itmx_rotation, about_point=itmx_position)        

        # ITMY
        itmy_position = np.array([x_end_prc_bs_1.get_value(), main_beam_y + 1.0, 0.0])
        itmy_rotation = PI/2
        self.itmy = Rectangle(
            color=BLUE, width=large_mirror_width, height=large_mirror_height, fill_opacity=0.5, stroke_width=2,
        ).move_to(itmy_position - large_mirror_width/2 * RIGHT).rotate(itmy_rotation, about_point=itmy_position)   

        # ETMX
        etmx_position = np.array([x_end_x_arm_1.get_value(), main_beam_y, 0.0])
        etmx_rotation = PI
        self.etmx = Rectangle(
            color=BLUE, width=large_mirror_width, height=large_mirror_height, fill_opacity=0.5, stroke_width=2,
        ).move_to(etmx_position - large_mirror_width/2 * RIGHT).rotate(etmx_rotation, about_point=etmx_position) 

        # ETMY
        etmy_position = np.array([x_end_prc_bs_1.get_value(), 3.0, 0.0])
        etmy_rotation = 3*PI/2
        self.etmy = Rectangle(
            color=BLUE, width=large_mirror_width, height=large_mirror_height, fill_opacity=0.5, stroke_width=2,
        ).move_to(etmy_position - large_mirror_width/2 * RIGHT).rotate(etmy_rotation, about_point=etmy_position)   


        ## Tex
        self.advanced_ligo_1 = Tex(
            "Advanced", 
            tex_template=TexFontTemplates.libertine
        ).scale(2).move_to(3*UP + 3*RIGHT)
        self.advanced_ligo_2 = Tex(
            "LIGO", 
            tex_template=TexFontTemplates.libertine
        ).scale(2).move_to(2*UP + 5.5*RIGHT) 

        self.ligo_definition_1 = Tex(
            "Laser"
        ).move_to(1*UP + 2*RIGHT)
        self.ligo_definition_2 = Tex(
            "Interferometer"
        ).next_to(self.ligo_definition_1, DOWN, aligned_edge=LEFT)
        self.ligo_definition_3 = Tex(
            "Gravitational-wave"
        ).next_to(self.ligo_definition_2, DOWN, aligned_edge=LEFT)
        self.ligo_definition_4 = Tex(
            "Observatory"
        ).next_to(self.ligo_definition_3, DOWN, aligned_edge=LEFT)

        self.core_question_1 = Tex(
            "Core question:",
            color=YELLOW
        ).scale(0.7).move_to(3.5*UP + 5.5*LEFT)
        self.core_question_2 = Tex(
            r"How does LIGO detect\\a gravitational wave?"
        ).scale(0.7).next_to(self.core_question_1, DOWN, aligned_edge=LEFT)

        self.core_answer_1 = Tex(
            r"LIGO acts as a {{ gravitational-wave }}\\to {{ laser power }} tranducer."
        ).scale(0.7).next_to(self.core_question_2, DOWN, aligned_edge=LEFT)
        self.core_answer_1[1].set_color(ORANGE)
        self.core_answer_1[3].set_color(RED)
        # This is to say, a gravitational wave interacts with the laser incident on the detector,
        # and produces a detectable laser signal corresponding to the gravitational wave at the output port.


        # "But how?", you ask.
        # "How can a gravitational wave be detectable?"
        # "Didn't you just say that gravitational waves are unfathomably small?"
        # "Aren't you claiming to use a laser with wavelength of 1 micron to detect a length change of 10^-20 meters?"
        # "Isn't you detector made of atoms which jiggle around at much greater distances than that?"
        
        # Core technology tex
        self.core_technology_tex_1 = Tex(
            "Core technology:",
            color=YELLOW
        ).scale(0.7).move_to(3.5*UP + 5.5*LEFT)
        self.core_technology_tex_2 = Tex(
            r"{{ Dual-recycled }}{{ Fabry P\'erot }}\\{{ Michelson }} interferometer"
        ).scale(0.7).next_to(self.core_technology_tex_1, DOWN, aligned_edge=LEFT)
        self.core_technology_tex_2[0].set_color(PURPLE)
        self.core_technology_tex_2[1].set_color(GREEN)
        self.core_technology_tex_2[3].set_color(RED)


        ## Core technology square
        ctr_width = x_end_x_arm_1.get_value() - x_start_prc_bs_1.get_value() + 1
        ctr_height = 7.0
        ctr_origin = np.array([(x_end_x_arm_1.get_value() + x_start_prc_bs_1.get_value())/2.0, 0.0, 0.0])
        self.core_tech_rect = Rectangle(
            width=ctr_width, height=ctr_height, color=YELLOW
        ).move_to(ctr_origin)

        ## LIGO logo
        logo_scale = 0.4
        logo_ring_width = 0.1
        number_of_rings = 6
        start_ring_angle = 3 * PI / 2
        ring_angle = PI / 2
        ring_color = DARK_GREY
        arc_center = 3.5 * UP + 4 * RIGHT

        rings = []
        for ii in range(1, number_of_rings + 1):
            outer_radius = ii * logo_scale
            inner_radius = ii * logo_scale - logo_ring_width
            ring = AnnularSector(
                inner_radius=inner_radius, outer_radius=outer_radius, start_angle=start_ring_angle, angle=ring_angle, arc_center=arc_center, color=ring_color
            )
            rings.append(ring)

        ## Updaters
        def time_updater(mob, dt):
            self.timer += dt * self.rate
            return

        def wave_updater(mob):
            temp_mob = self.make_electric_field(
                mob.amp, mob.k, mob.freq, self.timer, mob.phase, mob.start, mob.end, mob.number_of_lines, mob.color, 
                mob.amp_scaler, mob.k_scaler, mob.freq_scaler, mob.phase_offset, mob.x_start, mob.x_end, mob.y_start, mob.y_end, 
                mob.stroke_width, mob.rotate_angle, mob.rotate_point
            )
            mob.become(temp_mob)
            return 

        def standing_updater(mob):
            fmob = mob.field1
            bmob = mob.field2
            temp_fmob = self.make_electric_field(
                fmob.amp, fmob.k, fmob.freq, self.timer, fmob.phase, fmob.start, fmob.end, fmob.number_of_lines, fmob.color, 
                fmob.amp_scaler, fmob.k_scaler, fmob.freq_scaler, fmob.phase_offset, fmob.x_start, fmob.x_end, fmob.y_start, fmob.y_end, 
                fmob.stroke_width, fmob.rotate_angle, fmob.rotate_point
            )
            temp_bmob = self.make_electric_field(
                bmob.amp, bmob.k, bmob.freq, self.timer, bmob.phase, bmob.start, bmob.end, bmob.number_of_lines, bmob.color, 
                bmob.amp_scaler, bmob.k_scaler, bmob.freq_scaler, bmob.phase_offset, bmob.x_start, bmob.x_end, bmob.y_start, bmob.y_end, 
                bmob.stroke_width, bmob.rotate_angle, bmob.rotate_point
            )
            temp_mob = self.sum_electric_fields(temp_fmob, temp_bmob, color=mob.color, reverse=mob.reverse, y_axis=mob.y_axis, stroke_width=mob.stroke_width)
            mob.become(temp_mob)
            return 

        self.input_beam.add_updater(wave_updater)
        self.imc_1_beam.add_updater(wave_updater)
        self.imc_2_beam.add_updater(wave_updater)
        self.imc_3_beam.add_updater(wave_updater)
        self.post_imc_beam.add_updater(wave_updater)
        self.prc_bs_1_beam.add_updater(wave_updater)
        self.prc_bs_2_beam.add_updater(wave_updater)
        self.bs_y_1_beam.add_updater(wave_updater)
        self.bs_y_2_beam.add_updater(wave_updater)
        self.bs_x_1_beam.add_updater(wave_updater)
        self.bs_x_2_beam.add_updater(wave_updater)
        self.y_arm_1_beam.add_updater(wave_updater)
        self.y_arm_2_beam.add_updater(wave_updater)
        self.x_arm_1_beam.add_updater(wave_updater)
        self.x_arm_2_beam.add_updater(wave_updater)
        self.bs_src_1_beam.add_updater(wave_updater)
        self.bs_src_2_beam.add_updater(wave_updater)
        self.post_src_beam.add_updater(wave_updater)

        self.prc_bs_beam.add_updater(standing_updater)
        self.bs_y_beam.add_updater(standing_updater)
        self.bs_x_beam.add_updater(standing_updater)
        self.y_arm_beam.add_updater(standing_updater)
        self.x_arm_beam.add_updater(standing_updater)
        self.bs_src_beam.add_updater(standing_updater)

        ##################
        ###   Scene    ###
        ##################

        self.play(
            GrowFromCenter(self.mc1),
            GrowFromCenter(self.mc3),
            GrowFromCenter(self.mc2),
            GrowFromCenter(self.prm),
            GrowFromCenter(self.bs),
            GrowFromCenter(self.itmy),
            GrowFromCenter(self.itmx),
            GrowFromCenter(self.etmy),
            GrowFromCenter(self.etmx),
            GrowFromCenter(self.srm),
            GrowFromCenter(self.photodetector)
        )
        self.wait(1)

        self.play(
            Create(self.input_beam)
        )
        self.play(
            Create(self.imc_1_beam)
        )

        self.play(
            Create(self.imc_2_beam)
        )

        self.play(
            Create(self.imc_3_beam)
        )

        self.play(
            Create(self.post_imc_beam)
        )

        # self.play(
        #     Create(self.prc_bs_1_beam)
        # )
        # self.play(
        #     Create(self.prc_bs_2_beam)
        # )
        self.play(
            Create(self.prc_bs_beam)
        )
        # self.play(
        #     Create(self.bs_y_1_beam)
        # )
        # self.play(
        #     Create(self.bs_y_2_beam)
        # )
        self.play(
            Create(self.bs_y_beam),
            Create(self.bs_x_beam)
        )

        # self.play(
        #     Create(self.bs_x_1_beam)
        # )
        # self.play(
        #     Create(self.bs_x_2_beam)
        # )
        # self.play(
        #     Create(self.y_arm_1_beam)
        # )
        # self.play(
        #     Create(self.y_arm_2_beam)
        # )
        self.play(
            Create(self.y_arm_beam),
            Create(self.x_arm_beam),
            run_time=2
        )

        # self.play(
        #     Create(self.x_arm_1_beam)
        # )
        # self.play(
        #     Create(self.x_arm_2_beam)
        # )
        # self.play(
        #     Create(self.bs_src_1_beam)
        # )
        # self.play(
        #     Create(self.bs_src_2_beam)
        # )
        self.play(
            Create(self.bs_src_beam)
        )
        self.play(
            Create(self.post_src_beam)
        )

        self.input_beam.add_updater(time_updater)
        self.wait(12)

        self.input_beam.remove_updater(time_updater)
        self.wait(1)

        for ring in rings:
            self.play(
                FadeIn(ring)
            )
        
        self.wait(1)

        self.play(
            # Write(self.advanced_ligo_1),
            Write(self.advanced_ligo_2)
        )
        self.wait(1)

        self.play(
            Write(self.ligo_definition_1),
        )
        self.play(
            Write(self.ligo_definition_2),
        )
        self.play(
            Write(self.ligo_definition_3),
        )
        self.play(
            Write(self.ligo_definition_4),
        )

        self.wait(1)

        self.play(
            Unwrite(self.ligo_definition_1),
        )
        self.play(
            Unwrite(self.ligo_definition_2),
        )
        self.play(
            Unwrite(self.ligo_definition_3),
        )
        self.play(
            Unwrite(self.ligo_definition_4),
        )

        self.wait(1)

        # Core question
        self.play(
            Write(self.core_question_1)
        )
        self.play(
            Write(self.core_question_2)
        )
        self.wait(1)
        self.play(
            Write(self.core_answer_1)
        )

        self.wait(5)

        self.play(
            Unwrite(self.core_question_1),
            Unwrite(self.core_question_2),
            Unwrite(self.core_answer_1)
        )

        # Core technology
        self.play(
            Write(self.core_technology_tex_1)
        )
        self.wait(1)
        self.play(
            Create(self.core_tech_rect)
        )
        self.play(
            Write(self.core_technology_tex_2[0])
        )
        self.play(
            Write(self.core_technology_tex_2[1])
        )
        self.play(
            Write(self.core_technology_tex_2[2])
        )
        self.play(
            Write(self.core_technology_tex_2[3])
        )
        self.play(
            Write(self.core_technology_tex_2[4])
        )
        self.wait(5)

        self.play(
            Unwrite(self.core_technology_tex_1),
            Unwrite(self.core_technology_tex_2),
            Uncreate(self.core_tech_rect)
        )    
        self.wait(1)