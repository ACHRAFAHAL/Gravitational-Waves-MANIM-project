from manim import *

class Title1(Scene):
    def construct(self):
        intro_text = Text("Détection des Ondes Gravitationnelles", font_size=48)
        intro_text2 = Text("Projets VIRGO et LIGO", font_size=48).move_to(DOWN)
        self.play(Write(intro_text))
        self.play(Write(intro_text2))
        self.wait(1)
        self.play(FadeOut(intro_text[:12]), FadeOut(intro_text2), 
                  intro_text[12:].animate.to_edge(UP).shift(LEFT * 6).scale(0.7))
        self.wait(1)

class Title2(Scene): 
    def construct(self):
        intro_text = Text("Détection des Ondes Gravitationnelles", font_size=48)
        intro_text2 = Text("Projets VIRGO et LIGO", font_size=48).move_to(DOWN)
        intro_text3 = Text("Effet des", font_size=48).move_to(LEFT * 3.18)
        self.play(Write(intro_text))
        self.play(Write(intro_text2))
        self.wait(1)
        self.play(FadeOut(intro_text[:12]), FadeOut(intro_text2), 
                  intro_text[12:].animate.to_edge(UP).shift(LEFT * 6).scale(0.7))
        self.wait(1)      
        self.play(intro_text[12:].animate.move_to(ORIGIN).scale(1.3))
        self.wait(2)
        self.play(Write(intro_text3), intro_text[12:].animate.shift(RIGHT * 1.7))
        self.wait(1)
        group = VGroup(intro_text3, intro_text[12:])
        self.play(group.animate.to_edge(UP).shift(LEFT * 3.5).scale(0.6))
        self.wait(1)

class Title3(Scene):
    def construct(self):
        text4 = Tex("Interféromètre de Fabry-Perot").scale(1.3)
        self.play(Write(text4))
        self.wait(3)
        
class outro(Scene):
    def construct(self):
        text = Tex("MERCI").scale(1.5)
        self.play(Write(text))
        self.wait(5)
            