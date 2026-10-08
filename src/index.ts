console.time("Execution Time");
const fs = require("fs");
//const {GPU} = require('gpu.js');
//const gpu = new GPU();
//console.log("GPU.js is running on: ", gpu.mode);


const functions = { // Define a set of mathematical functions that get composited to create each node's activation function
    sigmoid: (x: number, a: number, b: number): number => {
        return a / (1 + Math.pow(b, -x));
    },
    tanh: (x: number, a: number, b: number): number => {
        return a * Math.tanh(b * x);
    },
    linear: (x: number, a: number, b: number): number => {
        return a * x + b;
    },
    logistic: (x: number, a: number, b: number): number => {
        return a / (1 + Math.exp(-b * x));
    },
    logarithmic: (x: number, a: number, b: number): number => {
        return a * Math.sign(x) * Math.log1p(Math.abs(b * x));
    },
    sin: (x: number, a: number, b: number): number => {
        return a * Math.sin(b * x);
    },
    cos: (x: number, a: number, b: number): number => {
        return a * Math.cos(b * x);
    },
    tan: (x: number, a: number, b: number): number => {
        return a * Math.tan(b * x);
    },
    sinh: (x: number, a: number, b: number): number => {
        return a * Math.sinh(b * x);
    },
    cosh: (x: number, a: number, b: number): number => {
        return a * Math.cosh(b * x);
    },
    pow: (x: number, a: number, b: number): number => {
        return a * (x ** b);
    },
    exp: (x: number, a: number, b: number): number => {
        return a * (b ** x);
    },
    quadratic: (x: number, a: number, b: number): number => {
        return a * (x ** 2) + b * x;
    },
    cubic: (x: number, a: number, b: number): number => {
        return a * (x ** 3) + b * (x ** 2) + x + (a*b)+a-b;
    },
    reciprocal: (x: number, a: number, b: number): number => {
        return a / (b + x);
    },
    absolute: (x: number, a: number, b: number): number => {
        return a * Math.abs(b * x);
    },
    step: (x: number, a: number, b: number): number => {
        return x < 0 ? a : b;
    }
};

type OperatingFunction = (x: number, a: number, b: number) => number;
type CompositeFunction = (x: number) => number;

function randomFunction(): (x: number, a: number, b: number) => number {
    const functionNames = Object.keys(functions);
    const randomIndex = Math.floor(Math.random() * functionNames.length);
    const randomFunctionName = functionNames[randomIndex];
    return functions[randomFunctionName as keyof typeof functions];
}

function randomPrimitiveFunction(): CompositeFunction {
    const operatingFunction: OperatingFunction = randomFunction();
    const a = Math.random();
    const b = Math.random() || Number.EPSILON;

    return (x: number): number => {
        return operatingFunction(x, a, b);
    };
}

function randomCompositeFunction(minFunctions = 2, maxFunctions = 5): OperatingFunction {
    const functionCount = Math.floor(Math.random() * (maxFunctions - minFunctions + 1)) + minFunctions;

    const composedFunctions: CompositeFunction[] = Array.from(
        { length: functionCount },
        () => randomPrimitiveFunction()
    );

    return (x: number): number => {
        return composedFunctions.reduce(
            (value, currentFunction) => currentFunction(value),
            x
        );
    };
}

class Node{
    input: number;
    output: number;
    outputConnections: Connection[];
    bias: number;
    a: number;
    b: number;
    operatingFunction: (x: number, a: number, b: number) => number;
    constructor(input: number, output: number, bias: number, operatingFunction: (x: number, a: number, b: number) => number){
        this.input = input;
        this.output = output;
        this.outputConnections = [];
        this.bias = bias;
        this.operatingFunction = operatingFunction;
        this.a = Math.random();
        this.b = Math.random();
    }
    addConnection(connection: Connection):void{
        this.outputConnections.push(connection);
    }
    getInput(input: number){
        this.input += input;
    }
    activate():void{
        this.output = this.operatingFunction(this.input, this.a, this.b) + this.bias;
    }
    passForward():void{
        if(this instanceof OutputNode){
            return;
        }
        for (let connection of this.outputConnections){
            connection.passOn(this.output);
        }
    }

}
class InputNode extends Node{
    constructor(){
        super(0, 0, 0, (x) => x);
    }
}
class OutputNode extends Node{
    constructor(bias:number, operatingFunction: (x: number, a: number, b: number) => number){
        super(0, 0, bias, operatingFunction);
    }
}
class Connection{
    inputNode: Node;
    outputNode: Node;
    weight: number;
    constructor(inputNode: Node, outputNode: Node, weight: number){
        this.inputNode = inputNode;
        this.outputNode = outputNode;
        this.weight = weight;
    }
    passOn(input: number):void{
        this.outputNode.input += input * this.weight;
        // this.outputNode.passForward(outputArray);
    }
}
class NeuralNetwork{
    Layers:Node[][];
    inputData: number[];
    error: number;
    constructor(inputData: number[], randomize: boolean = true){
        this.Layers = [];
        this.inputData = inputData;
        this.error = 0;
        if(randomize){
            const num_layers:number = Math.floor(Math.random()*10) + 2;
            for(let i = 0; i < num_layers; i++){
                    this.Layers.push([]);
            }
            for (let i = 0; i < num_layers; i++) {
                const num_nodes =
                    i === 0 || i === num_layers - 1
                        ? this.inputData.length
                        : Math.floor(Math.random()*10) + 1;

                for (let j = 0; j < num_nodes; j++) {
                    if (i === 0) {
                        this.Layers[i].push(new InputNode());
                    } else if (i === num_layers - 1) {
                        this.Layers[i].push(
                            new OutputNode(Math.random(), randomCompositeFunction())
                        );
                    } else {
                        this.Layers[i].push(
                            new Node(0, 0, Math.random(), randomCompositeFunction())
                        );
                    }
                }
            }
            for(let i = 0; i < num_layers - 1; i++){
                for(let j = 0; j < this.Layers[i].length; j++){
                    for(let k = 0; k < this.Layers[i+1].length; k++){
                        const weight:number = Math.random();
                        const connection:Connection = new Connection(this.Layers[i][j], this.Layers[i+1][k], weight);
                        this.Layers[i][j].addConnection(connection);
                    }
                }
            }
        }
    }
    neuralize():number[]{
        const outputArray:number[] = [];
        for (const layer of this.Layers) {
            for (const node of layer) {
                node.input = 0;
                node.output = 0;
            }
        }
        for(let i = 0; i < this.inputData.length; i++){
            this.Layers[0][i].input = this.inputData[i];
            this.Layers[0][i].passForward();
        }
        for(let i = 1; i < this.Layers.length; i++){
            for(let j = 0; j < this.Layers[i].length; j++){
                this.Layers[i][j].activate();
                this.Layers[i][j].passForward();
            }
        }
        /*for(let i = 1; i < this.Layers.length; i++){
            const kernel = gpu.createKernel(function(layer: Node[]){
                const node = layer[this.thread.x];
                node.activate();
                node.passForward();
            }).setOutput([this.Layers[i].length]);

            kernel(this.Layers[i]);
        }*/
        for(let i = 0; i < this.Layers[this.Layers.length - 1].length; i++){
            outputArray.push(this.Layers[this.Layers.length - 1][i].output);
        }
        return outputArray;
    }
    clone():NeuralNetwork{
        const clonedNetwork:NeuralNetwork = new NeuralNetwork(this.inputData, false);
        clonedNetwork.Layers = this.Layers.map(layer => layer.map(node => {
            const clonedNode:Node = new Node(node.input, node.output, node.bias, node.operatingFunction);
            clonedNode.a = node.a;
            clonedNode.b = node.b;
            return clonedNode;
        }));
        for(let i = 0; i < this.Layers.length - 1; i++){
            for(let j = 0; j < this.Layers[i].length; j++){
                for(let k = 0; k < this.Layers[i+1].length; k++){
                    const originalConnection:Connection = this.Layers[i][j].outputConnections[k];
                    const clonedConnection:Connection = new Connection(clonedNetwork.Layers[i][j], clonedNetwork.Layers[i+1][k], originalConnection.weight);
                    clonedNetwork.Layers[i][j].addConnection(clonedConnection);
                }
            }
        }
        return clonedNetwork;
    }
    setError(error: number):void{
        this.error = error;
    }
}

function trainNeuralNetwork(inputData: number[], outputData: number[], batchSize: number, reps: number): NeuralNetwork[] {
    let population = Array.from({ length: batchSize }, () => new NeuralNetwork(inputData));

    let bestNetwork: NeuralNetwork | null = null;
    let bestError: number = Infinity;

    for (let generation = 0; generation < reps; generation++) {
        for (const network of population) {
            const output = network.neuralize();

            const error = output.reduce((total, value, index) => total + Math.abs(value - outputData[index]), 0);

            network.setError(error);

            if (error < bestError) {
                bestError = error;
                bestNetwork = network.clone();
                bestNetwork.setError(error);
            }
        }

        population.sort((first, second) => first.error - second.error);

        const eliteCount = Math.min(3, batchSize, population.length);
        const nextPopulation = population
            .slice(0, eliteCount)
            .map(network => network.clone());

        while (nextPopulation.length < batchSize) {
            const parentIndex = Math.floor(Math.random() * eliteCount);
            const child = population[parentIndex].clone();
            mutateNetwork(child, generation, bestError);
            nextPopulation.push(child);
        }

        population = nextPopulation;
    }

    return bestNetwork ? [bestNetwork] : [];
}

function randomGaussian(): number {
    const u = 1 - Math.random();
    const v = 1 - Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function mutateNetwork(network: NeuralNetwork, generation: number, error: number): void {
    let mutationRate: number = (0.5)/(Math.log(generation + 10));
    for (const layer of network.Layers) {
        for (const node of layer) {
            if (node instanceof InputNode) {
                continue;
            }

            if (Math.random() < mutationRate) {
                node.bias += (Math.random() < 0.5 ? 1 : -1) * randomGaussian() * 0.1;
            }
            if (Math.random() < mutationRate) {
                node.a += (Math.random() < 0.5 ? 1 : -1) * randomGaussian() * 0.1;
            }
            if (Math.random() < mutationRate) {
                node.b += (Math.random() < 0.5 ? 1 : -1) * randomGaussian() * 0.1;
            }
            if (Math.random() < mutationRate/500) {
                node.operatingFunction = randomCompositeFunction();
            }
            for (const connection of node.outputConnections) {
                if (Math.random() < mutationRate) {
                    connection.weight += (Math.random() < 0.5 ? 1 : -1) * randomGaussian() * 0.1;
                }
            }
        }
    }
}

function main(): void {
    let inputData:number[] = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    let testArray:number[] = [2, 4, 6, 8, 10, 12, 14, 16, 18];
    let neuralNetworks:NeuralNetwork[] = trainNeuralNetwork(inputData, testArray, 850, 1500);
    console.log("Input Data: ", inputData);
    for(let neuralNetwork of neuralNetworks){
        let output:number[] = neuralNetwork.neuralize();
        console.log("Output Data: ", output);
        console.log("Error: ", neuralNetwork.error);
        console.log("Average Error: ", neuralNetwork.error/inputData.length);
        let totalPercentError = 0;
        for(let i = 0; i < output.length; i++){
            totalPercentError += Math.abs((output[i] - testArray[i]) / testArray[i]);
        }
        const averagePercentError = (totalPercentError / output.length) * 100;
        console.log("Average Percent Error: ", averagePercentError, "%");
        const jsonString = JSON.stringify(
            neuralNetwork,
            (key, value) =>
                key === "inputNode" || key === "outputNode" ? undefined : value,
            2
        );
        fs.writeFileSync("neural_network.json", jsonString);
    }
    console.log("Output Reference: ", testArray);
}

main();
console.timeEnd("Execution Time");