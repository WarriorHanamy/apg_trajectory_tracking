"""
2D rendering framework
"""

from __future__ import annotations

from typing import Any, Sequence

import math
import os
import sys

if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite
import pyglet

try:
    import pyglet
except ImportError:
    raise ImportError(
        """
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gymnasium dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gymnasium[all]' will do it.
    """
    )

try:
    from pyglet.gl import *
except ImportError:
    raise ImportError(
        """
    Error occurred while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    """
    )

import numpy as np
from gymnasium import error

from neural_control.environments.helper_simple_env import RenderImage

RAD2DEG = 57.29577951308232



def get_display(spec: str | None):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return pyglet.canvas.get_display()
        # returns already available pyglet_display,
        # if there is no pyglet display available then it creates one
    elif isinstance(spec, str):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            "Invalid display specification: {}. (Must be a string like :0 or None.)".format(
                spec
            )
        )


def get_window(
    width: int,
    height: int,
    display,
    **kwargs: Any,
) -> pyglet.window.Window:
    """
    Will create a pyglet window from the display specification provided.
    """
    screen = display.get_screens()  # available screens
    config = screen[0].get_best_config()  # selecting the first screen
    context = config.create_context(None)  # create GL context

    return pyglet.window.Window(
        width=width,
        height=height,
        display=display,
        config=config,
        context=context,
        **kwargs,
    )


class Viewer:
    def __init__(
        self,
        width: int,
        height: int,
        display: str | None = None,
    ) -> None:
        display_obj = get_display(display)

        self.width = width
        self.height = height
        self.window: pyglet.window.Window = get_window(
            width=width, height=height, display=display_obj
        )
        self.window.on_close = self.window_closed_by_user
        self.isopen: bool = True
        self.geoms: list[Geom] = []
        self.onetime_geoms: list[Geom] = []
        self.transform = Transform()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def close(self) -> None:
        if self.isopen and sys.meta_path:
            # ^^^ check sys.meta_path to avoid 'ImportError: sys.meta_path is None, Python is likely shutting down'
            self.window.close()
            self.isopen = False

    def window_closed_by_user(self) -> None:
        self.isopen = False

    def set_bounds(self, left: float, right: float, bottom: float, top: float) -> None:
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(
            translation=(-left * scalex, -bottom * scaley), scale=(scalex, scaley)
        )

    def add_geom(self, geom: "Geom") -> None:
        self.geoms.append(geom)

    def add_onetime(self, geom: "Geom") -> None:
        self.onetime_geoms.append(geom)

    def render(self, return_rgb_array: bool = False) -> RenderImage | bool:
        glClearColor(1, 1, 1, 1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        self.transform.enable()
        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()
        self.transform.disable()
        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            # In https://github.com/openai/gym-http-api/issues/2, we
            # discovered that someone using Xmonad on Arch was having
            # a window of size 598 x 398, though a 600 x 400 window
            # was requested. (Guess Xmonad was preserving a pixel for
            # the boundary.) So we use the buffer height/width rather
            # than the requested one.
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        self.window.flip()
        self.onetime_geoms = []
        return arr if return_rgb_array else self.isopen

    # Convenience
    def draw_circle(
        self,
        radius: float = 10,
        res: int = 30,
        filled: bool = True,
        **attrs: Any,
    ) -> "Geom":
        geom = make_circle(radius=radius, res=res, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polygon(
        self,
        v: Sequence[tuple[float, float]],
        filled: bool = True,
        **attrs: Any,
    ) -> "Geom":
        geom = make_polygon(v=v, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polyline(
        self,
        v: Sequence[tuple[float, float]],
        **attrs: Any,
    ) -> "Geom":
        geom = make_polyline(v=v)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_line(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        **attrs: Any,
    ) -> "Geom":
        geom = Line(start, end)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def get_array(self) -> RenderImage:
        self.window.flip()
        image_data = (
            pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        )
        self.window.flip()
        arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep="")
        arr = arr.reshape(self.height, self.width, 4)
        return arr[::-1, :, 0:3]

    def __del__(self) -> None:
        self.close()


def _add_attrs(geom: "Geom", attrs: dict[str, Any]) -> None:
    if "color" in attrs:
        geom.set_color(*attrs["color"])
    if "linewidth" in attrs:
        geom.set_linewidth(attrs["linewidth"])


class Geom:
    def __init__(self) -> None:
        self._color = Color((0, 0, 0, 1.0))
        self.attrs: list[Attr] = [self._color]

    def render(self) -> None:
        for attr in reversed(self.attrs):
            attr.enable()
        self.render1()
        for attr in self.attrs:
            attr.disable()

    def render1(self) -> None:
        raise NotImplementedError

    def add_attr(self, attr: "Attr") -> None:
        self.attrs.append(attr)

    def set_color(self, r: float, g: float, b: float) -> None:
        self._color.vec4 = (r, g, b, 1)


class Attr:
    def enable(self) -> None:
        raise NotImplementedError

    def disable(self) -> None:
        pass


class Transform(Attr):
    def __init__(
        self,
        translation: tuple[float, float] = (0.0, 0.0),
        rotation: float = 0.0,
        scale: tuple[float, float] = (1, 1),
    ) -> None:
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)

    def enable(self) -> None:
        glPushMatrix()
        glTranslatef(
            self.translation[0], self.translation[1], 0
        )  # translate to GL loc ppint
        glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)
        glScalef(self.scale[0], self.scale[1], 1)

    def disable(self) -> None:
        glPopMatrix()

    def set_translation(self, newx: float, newy: float) -> None:
        self.translation = (float(newx), float(newy))

    def set_rotation(self, new: float) -> None:
        self.rotation = float(new)

    def set_scale(self, newx: float, newy: float) -> None:
        self.scale = (float(newx), float(newy))


class Color(Attr):
    def __init__(self, vec4: tuple[float, float, float, float]) -> None:
        self.vec4 = vec4

    def enable(self) -> None:
        glColor4f(*self.vec4)


class LineStyle(Attr):
    def __init__(self, style: int) -> None:
        self.style = style

    def enable(self) -> None:
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(1, self.style)

    def disable(self) -> None:
        glDisable(GL_LINE_STIPPLE)


class LineWidth(Attr):
    def __init__(self, stroke: float) -> None:
        self.stroke = stroke

    def enable(self) -> None:
        glLineWidth(self.stroke)


class Point(Geom):
    def __init__(self) -> None:
        Geom.__init__(self)

    def render1(self) -> None:
        glBegin(GL_POINTS)  # draw point
        glVertex3f(0.0, 0.0, 0.0)
        glEnd()


class FilledPolygon(Geom):
    def __init__(self, v: Sequence[tuple[float, float]]) -> None:
        Geom.__init__(self)
        self.v = v

    def render1(self) -> None:
        if len(self.v) == 4:
            glBegin(GL_QUADS)
        elif len(self.v) > 4:
            glBegin(GL_POLYGON)
        else:
            glBegin(GL_TRIANGLES)
        for p in self.v:
            glVertex3f(p[0], p[1], 0)  # draw each vertex
        glEnd()


def make_circle(
    radius: float = 10,
    res: int = 30,
    filled: bool = True,
) -> Geom:
    points: list[tuple[float, float]] = []
    for i in range(res):
        ang = 2 * math.pi * i / res
        points.append((math.cos(ang) * radius, math.sin(ang) * radius))
    if filled:
        return FilledPolygon(points)
    else:
        return PolyLine(points, True)


def make_polygon(
    v: Sequence[tuple[float, float]],
    filled: bool = True,
) -> Geom:
    if filled:
        return FilledPolygon(v)
    else:
        return PolyLine(v, True)


def make_polyline(v: Sequence[tuple[float, float]]) -> Geom:
    return PolyLine(v, False)


def make_capsule(length: float, width: float) -> Geom:
    l, r, t, b = 0, length, width / 2, -width / 2
    box = make_polygon([(l, b), (l, t), (r, t), (r, b)])
    circ0 = make_circle(width / 2)
    circ1 = make_circle(width / 2)
    circ1.add_attr(Transform(translation=(length, 0)))
    geom = Compound([box, circ0, circ1])
    return geom


class Compound(Geom):
    def __init__(self, gs: Sequence[Geom]) -> None:
        Geom.__init__(self)
        self.gs = list(gs)
        for g in self.gs:
            g.attrs = [a for a in g.attrs if not isinstance(a, Color)]

    def render1(self) -> None:
        for g in self.gs:
            g.render()


class PolyLine(Geom):
    def __init__(self, v: Sequence[tuple[float, float]], close: bool) -> None:
        Geom.__init__(self)
        self.v = v
        self.close = close
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self) -> None:
        glBegin(GL_LINE_LOOP if self.close else GL_LINE_STRIP)
        for p in self.v:
            glVertex3f(p[0], p[1], 0)  # draw each vertex
        glEnd()

    def set_linewidth(self, x: float) -> None:
        self.linewidth.stroke = x


class Line(Geom):
    def __init__(
        self,
        start: tuple[float, float] = (0.0, 0.0),
        end: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        Geom.__init__(self)
        self.start = start
        self.end = end
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self) -> None:
        glBegin(GL_LINES)
        glVertex2f(*self.start)
        glVertex2f(*self.end)
        glEnd()


class Image(Geom):
    def __init__(self, fname: str, width: float, height: float) -> None:
        Geom.__init__(self)
        self.set_color(1.0, 1.0, 1.0)
        self.width = width
        self.height = height
        img = pyglet.image.load(fname)
        self.img = img
        self.flip = False

    def render1(self) -> None:
        self.img.blit(
            -self.width / 2, -self.height / 2, width=self.width, height=self.height
        )


# ================================================================


class SimpleImageViewer:
    def __init__(self, display: str | None = None, maxwidth: int = 500) -> None:
        self.window: pyglet.window.Window | None = None
        self.isopen = False
        self.display = get_display(display)
        self.maxwidth = maxwidth

    def imshow(self, arr: RenderImage) -> None:
        if self.window is None:
            height, width, _channels = arr.shape
            if width > self.maxwidth:
                scale = self.maxwidth / width
                width = int(scale * width)
                height = int(scale * height)
            self.window = get_window(
                width=width,
                height=height,
                display=self.display,
                vsync=False,
                resizable=True,
            )
            self.width = width
            self.height = height
            self.isopen = True

            @self.window.event
            def on_resize(width, height):
                self.width = width
                self.height = height

            @self.window.event
            def on_close():
                self.isopen = False

        assert len(arr.shape) == 3, "You passed in an image with the wrong number shape"
        image = pyglet.image.ImageData(
            arr.shape[1], arr.shape[0], "RGB", arr.tobytes(), pitch=arr.shape[1] * -3
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        texture = image.get_texture()
        texture.width = self.width
        texture.height = self.height
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        texture.blit(0, 0)  # draw
        self.window.flip()

    def close(self) -> None:
        if self.isopen and sys.meta_path:
            # ^^^ check sys.meta_path to avoid 'ImportError: sys.meta_path is None, Python is likely shutting down'
            self.window.close()
            self.isopen = False

    def __del__(self) -> None:
        self.close()
