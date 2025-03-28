package src;

import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class iragarpenakEgin {
    private static void main(Instances dataset) {
        LinearRegression modelLR = linearRegression.linearRegressionSortu(null);
        if (modelLR != null) {
            linearRegressionIragarri(modelLR, dataset);
        }
    }
    
    private static void linearRegressionIragarri(LinearRegression model, Instances dataset) {
        String outputFilePath = "/home/ibai/GitHub/weka-project/probaData/iragarpena_LinearRegression.txt";
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath))) {
            for (int i = 0; i < dataset.numInstances(); i++) {
                Instance instance = dataset.instance(i);
                double prediction = model.classifyInstance(instance);
                String predictedClass = prediction > 0.5 ? "Pos" : "Neg";
                writer.write("Instance " + (i + 1) + ": Predicted Class = " + predictedClass + "\n");
            }
            System.out.println("Predictions saved to: " + outputFilePath);
        } catch (IOException e) {
            System.out.println("ERROREA: Ezin izan da iragarpenak fitxategian gorde.");
            e.printStackTrace();
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da iragarpenik egin.");
            e.printStackTrace();
        }
    }
}