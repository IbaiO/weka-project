package src;

import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;

public class iragarpenakEgin {
    private static void main(Instances dataset) {
        LinearRegression modelLR = linearRegression.linearRegressionSortu(null);
        if (modelLR != null) {
            linearRegressionIragarri(modelLR, dataset);
        }
    }
    
    private static void linearRegressionIragarri(LinearRegression model, Instances dataset) {
        try {
            for (int i = 0; i < dataset.numInstances(); i++) {
                Instance instance = dataset.instance(i);
                double prediction = model.classifyInstance(instance);
                String predictedClass = prediction > 0.5 ? "Pos" : "Neg";
                System.out.println("Predicted class for instance " + (i + 1) + ": " + predictedClass);
            }
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da iragarpenik egin.");
            e.printStackTrace();
        }
    }
    
    //hemen ipintzeko iragarpenak egiteko klasea baina orokortua, en vez de bakoitza bere classifier klasean
}
