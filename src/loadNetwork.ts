import { loadNetwork } from './NeuralNetwork.ts';

const network = await loadNetwork('./neural_network.json');
console.log(network.neuralize());