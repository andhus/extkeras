from __future__ import division, print_function

from collections import OrderedDict


class ChildrenLayersMixin(object):
    """Mixin for using internal layers in arbitrary ways internally in a Layer.

    All internal layers should be:
    - appended to self.children
    - built in build of layer

    Should be inherited first!

    TODO complete docs.
    """
    @property
    def children(self):
        if not hasattr(self, '_children'):
            return OrderedDict([])
        else:
            return self._children

    def add_child(self, identifier, layer):
        if not hasattr(self, '_children'):
            self._children = OrderedDict([])
        self._children[identifier] = layer

        return layer

    @property
    def trainable_weights(self):
        if self.trainable:
            return self._trainable_weights + sum(
                [child.trainable_weights for child in self.children.values()],
                []
            )
        else:
            return []

    @property
    def non_trainable_weights(self):
        children_non_trainable = sum(
            [child.non_trainable_weights for child in self.children.values()],
            []
        )
        if self.trainable:
            return self._non_trainable_weights + children_non_trainable
        else:
            return (
                self._trainable_weights +
                self._non_trainable_weights +
                children_non_trainable
            )

    @property
    def trainable_variables(self):
        return self.trainable_weights

    @property
    def non_trainable_variables(self):
        return self.non_trainable_weights

    @property
    def weights(self):
        """Returns the list of all layer variables/weights.

        Returns:
          A list of variables.
        """
        return self.trainable_weights + self.non_trainable_weights

    @property
    def variables(self):
        """Returns the list of all layer variables/weights.

        Returns:
          A list of variables.
        """
        return self.weights

    # TODO implement get/set_weights!
