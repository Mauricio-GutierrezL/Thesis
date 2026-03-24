# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pennylane as qml
import numpy as np
import jax
from sklearn.base import BaseEstimator, ClassifierMixin
from qml_benchmarks.model_utils import *

jax.config.update("jax_enable_x64", True)

#NEW 
class CircuitCentricClassifier(ClassifierMixin, BaseEstimator):
#ENDNEW
    def __init__(
        self,
        n_input_copies=2, 
        n_layers=4, #single quebit rotation +entangling

        # NEW:
        ansatz_type="strongly",          # "strongly" (original) or "controlled"
        entanglement_level=1.0,          # used only if ansatz_type="controlled"
        entangler="CNOT",                # impritive
        # ENDNEW
        convergence_interval=200,
        max_steps=10000,
        learning_rate=0.001,
        batch_size=32,
        max_vmap=None,
        jit=True,
        scaling=1.0,
        random_state=42,
        dev_type="default.qubit",
        qnode_kwargs={"interface": "jax-jit"},
    ):
        r"""
        A variational quantum model based on https://arxiv.org/pdf/1804.03680v2.pdf.

        Uses amplitude encoding as the data embedding, but implements several
        copies of the initial state. The data vector :math:`x` of size :math:`N`
        gets padded to the next power of 2 by values :math:`1/len(\tilde{x})`
        (so that the new data vector :math:`\tilde{x}` has length :math:`2^n > N`).
        The padded vector gets normalised to :math:`\bar{\tilde{x}}`, and
        embedded into the amplitudes of several copies of :math:`n`-qubit
        quantum states. Altogether, the variational circuit acts on a quantum
        state of amplitudes :math:`\bar{\tilde{x}} \otimes \bar{\tilde{x}} \otimes ...`.

        The total number of qubits :math:`d\lceil \log(n_features) \rceil` of
        this model depends on the number of features :math:`N` and the number
        of copies :math:`d`.

        The variational part of the circuit uses general single qubit rotations
        as trainable gates. Each layer in the ansatz first applies a rotation
        to each qubit, and then controlled rotations connecting each qubit
        :math:`i` to qubit :math:`i+r \text{mod } n`, where :math:`r` repeatedly runs
        through the range :math:`[0,...,n-1]`. The number of layers in the
        ansatz is a hyperparameter of the model.

        The result of the model is the sigma-z expectation of the first qubit,
        plus a trainable scalar bias. Training is via the square loss.

        Args:
            n_input_copies (int): Number of copies of the amplitude embedded
                state to produce.
            n_layers (int): Number of layers in the variational ansatz.
            convergence_interval (int): The number of loss values to consider to decide convergence.
            max_steps (int): Maximum number of training steps. A warning will
                be raised if training did not converge.
            learning_rate (float): Initial learning rate for gradient descent.
            batch_size (int): Size of batches used for computing parameter updates.
            max_vmap (int or None): The maximum size of a chunk to vectorise over. Lower values use less memory.
                must divide batch_size.
            jit (bool): Whether to use just in time compilation.
            dev_type (str): Pennylane device type; e.g. 'default.qubit'.
            qnode_kwargs (str): Keyword arguments for the circuit qnode.
            scaling (float): Factor by which to scale the input data.
            random_state (int): Seed used for pseudorandom number generation.
        """
        # NEW 
        # int typecasted
        self.n_input_copies = int(n_input_copies)
        self.n_layers = int(n_layers)
        self.convergence_interval = int(convergence_interval)
        self.max_steps = int(max_steps)
        self.batch_size = int(batch_size)
        # ENDNEW
        # attributes that do not depend on data
       
        self.learning_rate = learning_rate
        self.dev_type = dev_type
        self.qnode_kwargs = qnode_kwargs
        self.jit = jit
        self.scaling = scaling
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        # NEW 
        # int typecasted
        if max_vmap is None:
            self.max_vmap = int(self.batch_size)
        else:
            self.max_vmap = int(max_vmap)
         # ENDNEW

        # data-dependant attributes
        # which will be initialised by calling "fit"
        self.params_ = None  # Dictionary containing the trainable parameters
        self.n_qubits_ = None
        self.scaler = None
        self.circuit = None

        #NEW
        self.ansatz_type = ansatz_type
        self.entanglement_level = float(entanglement_level)
        self.entangler = entangler

    def generate_key(self) -> jax.Array:
        """
        Generates a random key used in sampling batches.

        Returns:
            jax.Array: _description_
        """
        return jax.random.PRNGKey(self.rng.integers(1000000))
    
    # NEW
    def _entangler_gate(self): # Correct entangler
        if self.entangler.upper() == "CNOT":
            return qml.CNOT
        if self.entangler.upper() == "CZ":
            return qml.CZ
        raise ValueError("entangler must be 'CNOT' or 'CZ'")

    def _controlled_ansatz(self, weights):
        """weights shape: (n_layers, n_qubits, 3). Applies Rot + partial ring entanglers."""
        gate = self._entangler_gate() 
        M = self.n_qubits_

        for l in range(self.n_layers):
            # 1) trainable single-qubit rotations
            for i in range(M):
                qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)

            # 2) ring edges (cycle range like StronglyEntanglingLayers-style mixing)
            if M > 1:
                r = (l % (M - 1)) + 1   # 1,2,3,...,M-1,1,2,...
                edges = [(i, (i + r) % M) for i in range(M)]

                k = int(round(self.entanglement_level * len(edges)))  # 0..M

                if k > 0:
                    # seeded, deterministic per layer
                    # run_hyperparameter_search.py, GridSearchCV creates models using the default random_state=42
                    # CV folds will use the same random_state
                    rng = np.random.default_rng(self.random_state + 1000 * l)
                    # randomstate for reproducibility, l for each other layer random
                    idx = rng.choice(len(edges), size=k, replace=False)
                    # select k different out of it
                    idx.sort()  # stable ordering (optional but nice)

                    for j in idx:
                        a, b = edges[j]
                        gate(wires=[a, b])
    #ENDNEW

    def construct_model(self):

        dev = qml.device(self.dev_type, wires=self.n_qubits_)

        @qml.qnode(dev, **self.qnode_kwargs)
        def circuit(params, x):
            # classically compute the copies of inputs
            # since AmplitudeEmbedding cannot be called
            # multiple times on jax device
            tensor_x = x
            for i in range(self.n_input_copies - 1):
                tensor_x = jnp.kron(tensor_x, x)

            qml.AmplitudeEmbedding(tensor_x, wires=range(self.n_qubits_))

            # NEW
            if self.ansatz_type == "strongly":
                qml.StronglyEntanglingLayers(params["weights"], wires=range(self.n_qubits_))
            else:
                self._controlled_ansatz(params["weights"])
            # ENDNEW
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

        if self.jit:
            circuit = jax.jit(circuit)

        def circuit_plus_bias(params, x):
            return circuit(params, x) + params["b"]

        if self.jit:
            circuit_plus_bias = jax.jit(circuit_plus_bias)
        self.forward = jax.vmap(circuit_plus_bias, in_axes=(None, 0))
        self.chunked_forward = chunk_vmapped_fn(self.forward, 1, self.max_vmap)

        return self.forward

    def initialize(self, n_features, classes=None):
        """Initialize attributes that depend on the number of features and the class labels.
        Args:
            n_features (int): Number of features that the classifier expects
            classes (array-like): class labels
        """
        if classes is None:
            classes = [-1, 1]

        self.classes_ = classes
        self.n_classes_ = len(self.classes_)
        assert self.n_classes_ == 2
        assert 1 in self.classes_ and -1 in self.classes_

        n_qubits_per_copy = int(np.ceil(np.log2(n_features)))
        self.n_qubits_ = self.n_input_copies * n_qubits_per_copy

        self.construct_model()
        self.initialize_params()

    def initialize_params(self):

        # NEW 
        if self.ansatz_type == "strongly": 
            shape = qml.StronglyEntanglingLayers.shape(
                n_layers=self.n_layers, n_wires=self.n_qubits_
            )
        else:
            shape = (self.n_layers, self.n_qubits_, 3)  # Rot params
        # ENDNEW
        weights = jax.random.uniform(
            self.generate_key(), minval=0, maxval=2 * np.pi, shape=shape
        )
        b = jnp.array(0.01)
        self.params_ = {"weights": weights, "b": b}


    def fit(self, X, y):
        """Fit the model to data X and labels y.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)
            y (np.ndarray): Labels of shape (n_samples,)
        """

        self.initialize(n_features=X.shape[1], classes=np.unique(y))

        X = self.transform(X)

        optimizer = optax.adam

        def loss_fn(params, X, y):
            pred = self.forward(
                params, X
            )  # jnp.stack([self.forward(params, x) for x in X])
            return jnp.mean(optax.l2_loss(pred, y))

        if self.jit:
            loss_fn = jax.jit(loss_fn)
        self.params_ = train(
            self,
            loss_fn,
            optimizer,
            X,
            y,
            self.generate_key,
            convergence_interval=self.convergence_interval,
        )

        return self

    def predict(self, X):
        """Predict labels for batch of data X.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)
        """
        predictions = self.predict_proba(X)
        mapped_predictions = np.argmax(predictions, axis=1)
        return np.take(self.classes_, mapped_predictions)

    def predict_proba(self, X):
        """Predict label probabilities for data X.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)

        Returns:
            y_pred_proba (np.ndarray): Predicted label probabilities of shape
            (n_samples, n_classes)
        """
        X = self.transform(X)
        predictions = self.chunked_forward(self.params_, X)
        predictions_2d = np.c_[(1 - predictions) / 2, (1 + predictions) / 2]
        return predictions_2d

    def transform(self, X, preprocess=False):
        """
        The feature vectors padded to the next power of 2 and then normalised.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)
        """
        n_features = X.shape[1]
        X = X * self.scaling

        n_qubits_per_copy = int(np.ceil(np.log2(n_features)))
        max_n_features = 2**n_qubits_per_copy
        n_padding = max_n_features - n_features
        padding = np.ones(shape=(len(X), n_padding)) / max_n_features

        X_padded = np.c_[X, padding]
        X_normalised = np.divide(
            X_padded, np.expand_dims(np.linalg.norm(X_padded, axis=1), axis=1)
        )
        return X_normalised
    

# NEW

class CircuitCentricClassifierHalfSeparable(CircuitCentricClassifier): 
    """CCC with ~50% of ring entanglers per layer (controlled ansatz)."""
    def __init__(
        self,
        n_input_copies=2,
        n_layers=4,
        ansatz_type="strongly",          # will be overridden
        entanglement_level=1.0,          # will be overridden
        entangler="CNOT",
        convergence_interval=200,
        max_steps=10000,
        learning_rate=0.001,
        batch_size=32,
        max_vmap=None,
        jit=True,
        scaling=1.0,
        random_state=42,
        dev_type="default.qubit",
        qnode_kwargs={"interface": "jax-jit"},
    ):
        super().__init__(
            n_input_copies=n_input_copies,
            n_layers=n_layers,
            ansatz_type="controlled",
            entanglement_level=0.5,
            entangler=entangler,
            convergence_interval=convergence_interval,
            max_steps=max_steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_vmap=max_vmap,
            jit=jit,
            scaling=scaling,
            random_state=random_state,
            dev_type=dev_type,
            qnode_kwargs=qnode_kwargs,
        )


class CircuitCentricClassifierSeparable(CircuitCentricClassifier):
    """CCC with 0 entangling gates in the ansatz (still uses amplitude embedding)."""
    def __init__(
        self,
        n_input_copies=2,
        n_layers=4,
        ansatz_type="strongly",          # will be overridden
        entanglement_level=1.0,          # will be overridden
        entangler="CNOT",
        convergence_interval=200,
        max_steps=10000,
        learning_rate=0.001,
        batch_size=32,
        max_vmap=None,
        jit=True,
        scaling=1.0,
        random_state=42,
        dev_type="default.qubit",
        qnode_kwargs={"interface": "jax-jit"},
    ):
        super().__init__(
            n_input_copies=n_input_copies,
            n_layers=n_layers,
            ansatz_type="controlled",
            entanglement_level=0.0,
            entangler=entangler,
            convergence_interval=convergence_interval,
            max_steps=max_steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_vmap=max_vmap,
            jit=jit,
            scaling=scaling,
            random_state=random_state,
            dev_type=dev_type,
            qnode_kwargs=qnode_kwargs,
        )
# ENDNEW