import * as fs from "node:fs";
import { readFile } from 'node:fs/promises';

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
    exp: (x: number, a: number, b: number): number => {
        return a * (b ** x);
    },
    mononomial: (x: number, a: number, b: number): number => {
        return a * (x ** b)
    },
    reciprocal: (x: number, a: number, b: number): number => {
        return a / (b + x);
    },
    absolute: (x: number, a: number, b: number): number => {
        return a * Math.abs(b * x);
    },
    step: (x: number, a: number, b: number): number => {
        return x < 0 ? a : b;
    },
    nat_exp: (x: number, a: number, b: number): number => {
        return Math.E ** (b * x + a);
    },
    constant: (x: number, a: number, b: number): number => {
        return a * b + 0 * x;
    },
    increment: (x: number, a: number, b: number): number => {
        return x + a * b;
    }
};

type ActivationPrimitive = {
    name: keyof typeof functions;
    a: number;
    b: number;
} | {
    name: "identity";
};
type OperatingFunction = ((x: number, a: number, b: number) => number) & {
    activationFunction?: ActivationPrimitive[];
};
type CompositeOperatingFunction = OperatingFunction & {
    activationFunction: ActivationPrimitive[];
};

type SerializedNode = {
    input: number;
    output: number;
    outputConnections: { weight: number }[];
    bias: number;
    a: number;
    b: number;
    activationFunction: ActivationPrimitive[];
};

type SerializedNetwork = {
    Layers: SerializedNode[][];
    inputData: number[];
    error: number;
};

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
    return typeof value === "number" && Number.isFinite(value);
}

function isActivationFunction(value: unknown): value is ActivationPrimitive[] {
    return Array.isArray(value) && value.every((primitive: unknown) => {
        if (!isRecord(primitive) || typeof primitive.name !== "string") {
            return false;
        }
        if (primitive.name === "identity") {
            return true;
        }
        return Object.hasOwn(functions, primitive.name)
            && isFiniteNumber(primitive.a)
            && isFiniteNumber(primitive.b);
    });
}

function isSerializedNetwork(value: unknown): value is SerializedNetwork {
    if (!isRecord(value)
        || !Array.isArray(value.inputData)
        || !value.inputData.every(isFiniteNumber)
        || !isFiniteNumber(value.error)
        || !Array.isArray(value.Layers)
        || value.Layers.length < 2
        || !value.Layers.every((layer: unknown) => Array.isArray(layer) && layer.length > 0)
        || value.Layers[0].length !== value.inputData.length) {
        return false;
    }

    return value.Layers.every((layer: unknown) =>
        Array.isArray(layer) && layer.every((node: unknown) =>
            isRecord(node)
            && isFiniteNumber(node.input)
            && isFiniteNumber(node.output)
            && isFiniteNumber(node.bias)
            && isFiniteNumber(node.a)
            && isFiniteNumber(node.b)
            && isActivationFunction(node.activationFunction)
            && Array.isArray(node.outputConnections)
            && node.outputConnections.every((connection: unknown) =>
                isRecord(connection) && isFiniteNumber(connection.weight)
            )
        )
    );
}

function operatingFunctionFromActivation(
    activationFunction: ActivationPrimitive[]
): CompositeOperatingFunction {
    const operatingFunction: CompositeOperatingFunction = (x: number): number =>
        activationFunction.reduce(
            (value, primitive) => primitive.name === "identity"
                ? value
                : functions[primitive.name](value, primitive.a, primitive.b),
            x
        );
    operatingFunction.activationFunction = activationFunction;
    return operatingFunction;
}

function randomFunction(): [keyof typeof functions, OperatingFunction] {
    const functionNames = Object.keys(functions);
    const randomIndex = Math.floor(Math.random() * functionNames.length);
    const randomFunctionName = functionNames[randomIndex] as keyof typeof functions;
    return [randomFunctionName, functions[randomFunctionName]];
}

function randomPrimitiveFunction(): ActivationPrimitive {
    const [name] = randomFunction();
    const a = Math.random()*100;
    const b = Math.random()*100 || Number.EPSILON;

    return { name, a, b };
}

function describeCompositeFunction(activationFunction: ActivationPrimitive[]): string {
    const expression = activationFunction.reduce((input, primitive) => {
        if (primitive.name === "identity") {
            return input;
        }
        return `functions[${JSON.stringify(primitive.name)}](${input}, ${primitive.a}, ${primitive.b})`;
    }, "x");

    return `x => ${expression}`;
}

function randomCompositeFunction(minFunctions = 2, maxFunctions = 5): CompositeOperatingFunction {
    const functionCount = Math.floor(Math.random() * (maxFunctions - minFunctions + 1)) + minFunctions;

    const activationFunction: ActivationPrimitive[] = Array.from(
        { length: functionCount },
        () => randomPrimitiveFunction()
    );

    const compositeFunction: CompositeOperatingFunction = (x: number): number => {
        return activationFunction.reduce(
            (value, primitive) => primitive.name === "identity"
                ? value
                : functions[primitive.name](value, primitive.a, primitive.b),
            x
        );
    };
    compositeFunction.activationFunction = activationFunction;
    return compositeFunction;
}

export class Node{
    input: number;
    output: number;
    outputConnections: Connection[];
    bias: number;
    a: number;
    b: number;
    operatingFunction: (x: number, a: number, b: number) => number;
    activationFunction: ActivationPrimitive[];
    activationFunctionExpression: string;
    constructor(input: number, output: number, bias: number, operatingFunction: OperatingFunction, activationFunction?: ActivationPrimitive[]){
        this.input = input;
        this.output = output;
        this.outputConnections = [];
        this.bias = bias;
        this.operatingFunction = operatingFunction;
        this.activationFunction = activationFunction ?? operatingFunction.activationFunction ?? [{ name: "identity" }];
        this.activationFunctionExpression = describeCompositeFunction(this.activationFunction);
        this.a = Math.random();
        this.b = Math.random();
    }
    setOperatingFunction(operatingFunction: OperatingFunction): void {
        this.operatingFunction = operatingFunction;
        this.activationFunction = operatingFunction.activationFunction ?? [{ name: "identity" }];
        this.activationFunctionExpression = describeCompositeFunction(this.activationFunction);
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
export class InputNode extends Node{
    constructor(){
        super(0, 0, 0, (x) => x);
    }
}
export class OutputNode extends Node{
    constructor(bias:number, operatingFunction: (x: number, a: number, b: number) => number){
        super(0, 0, bias, operatingFunction);
    }
}
export class Connection{
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
export class NeuralNetwork{
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
        for(let i = 0; i < this.Layers[this.Layers.length - 1].length; i++){
            outputArray.push(this.Layers[this.Layers.length - 1][i].output);
        }
        return outputArray;
    }
    clone():NeuralNetwork{
        const clonedNetwork:NeuralNetwork = new NeuralNetwork(this.inputData, false);
        clonedNetwork.Layers = this.Layers.map(layer => layer.map(node => {
            const clonedNode:Node = new Node(
                node.input,
                node.output,
                node.bias,
                node.operatingFunction,
                node.activationFunction
            );
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


export function trainNeuralNetwork(inputData: number[], outputData: number[], batchSize: number, reps: number): NeuralNetwork[] {
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

export function mutateNetwork(network: NeuralNetwork, generation: number, error: number): void {
    let mutationRate: number = (0.5)/(Math.log(generation <= 10 ? 11 : generation + 1));
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
            if (Math.random() < (mutationRate > 0.001 && error > 0.05 ? mutationRate : mutationRate/500)) {
                node.setOperatingFunction(randomCompositeFunction());
            }
            for (const connection of node.outputConnections) {
                if (Math.random() < mutationRate) {
                    connection.weight += (Math.random() < 0.5 ? 1 : -1) * randomGaussian() * 0.1;
                }
            }
        }
    }
}

export async function loadNetwork(fname: string): Promise<NeuralNetwork> {
    const json = await readFile(fname, "utf8");
    const parsed: unknown = JSON.parse(json);
    if (!isSerializedNetwork(parsed)) {
        throw new TypeError(`Invalid neural network JSON in ${fname}`);
    }

    for (let layerIndex = 0; layerIndex < parsed.Layers.length; layerIndex++) {
        const layer = parsed.Layers[layerIndex];
        const expectedConnections = parsed.Layers[layerIndex + 1]?.length ?? 0;
        for (const node of layer) {
            if (node.outputConnections.length !== expectedConnections) {
                throw new TypeError(`Invalid connection count in layer ${layerIndex} of ${fname}`);
            }
        }
    }

    const network = new NeuralNetwork(parsed.inputData, false);
    network.Layers = parsed.Layers.map((layer, layerIndex) =>
        layer.map((serializedNode) => {
            const operatingFunction = operatingFunctionFromActivation(serializedNode.activationFunction);
            const node = layerIndex === 0
                ? new InputNode()
                : layerIndex === parsed.Layers.length - 1
                    ? new OutputNode(serializedNode.bias, operatingFunction)
                    : new Node(serializedNode.input, serializedNode.output, serializedNode.bias, operatingFunction);

            node.input = serializedNode.input;
            node.output = serializedNode.output;
            node.bias = serializedNode.bias;
            node.a = serializedNode.a;
            node.b = serializedNode.b;
            node.activationFunction = serializedNode.activationFunction;
            node.activationFunctionExpression = describeCompositeFunction(serializedNode.activationFunction);
            return node;
        })
    );

    for (let layerIndex = 0; layerIndex < network.Layers.length - 1; layerIndex++) {
        for (let nodeIndex = 0; nodeIndex < network.Layers[layerIndex].length; nodeIndex++) {
            const sourceNode = network.Layers[layerIndex][nodeIndex];
            const serializedNode = parsed.Layers[layerIndex][nodeIndex];
            for (let connectionIndex = 0; connectionIndex < serializedNode.outputConnections.length; connectionIndex++) {
                sourceNode.addConnection(new Connection(
                    sourceNode,
                    network.Layers[layerIndex + 1][connectionIndex],
                    serializedNode.outputConnections[connectionIndex].weight
                ));
            }
        }
    }

    network.setError(parsed.error);
    return network;
}