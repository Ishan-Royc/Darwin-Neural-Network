console.time("Execution Time");
import * as NeuralNetwork from './NeuralNetwork.ts';
import * as fs from "node:fs";


function main(): void {
    let inputData:number[] = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    let testArray:number[] = [2, 4, 6, 8, 10, 12, 14, 16, 18];
    let neuralNetworks:NeuralNetwork.NeuralNetwork[] = NeuralNetwork.trainNeuralNetwork(inputData, testArray, 850, 1500);
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