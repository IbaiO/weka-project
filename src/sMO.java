package src;

import weka.classifiers.functions.SMO;
import weka.classifiers.Evaluation;

import weka.core.Instances;
import java.util.Random;

public class sMO {

    public static SMO[] main(Instances dataset) throws Exception {
    
        // Ignorar advertencias de netlib-java y establecer propiedades para usar F2jBLAS y F2jLAPACK
        System.setProperty("com.github.fommil.netlib.BLAS", "com.github.fommil.netlib.F2jBLAS");
        System.setProperty("com.github.fommil.netlib.LAPACK", "com.github.fommil.netlib.F2jLAPACK");

        if (dataset == null) {
            System.out.println("Error: Unable to load training data.");
            return null;
        }

        // Verificar si el atributo de clase es nominal o numérico
        if (dataset.classAttribute().isNominal()) {
            System.out.println("The class attribute is nominal.");
        } else if (dataset.classAttribute().isNumeric()) {
            System.out.println("The class attribute is numeric.");
        } else {
            System.out.println("The class attribute is neither nominal nor numeric.");
        }
        dataset.setClassIndex(0); // Set the class attribute index
        // Comparar diferentes kernels y optimizar C
        System.out.println("Evaluating PolyKernel...");
        SMO polyKernelModel = evaluateKernel(dataset, new weka.classifiers.functions.supportVector.PolyKernel(), 2); // E = 2

        System.out.println("Evaluating RBFKernel...");
        SMO rbfKernelModel = evaluateKernel(dataset, new weka.classifiers.functions.supportVector.RBFKernel(), 0.01); // Gamma = 0.1

        System.out.println("Evaluating PukKernel...");
        SMO pukKernelModel = evaluateKernel(dataset, new weka.classifiers.functions.supportVector.Puk(), 1.0); // Omega = 1.0


        // Crear un arreglo para almacenar los modelos evaluados
        SMO[] models = {polyKernelModel, rbfKernelModel, pukKernelModel};

        // Devolver ambos modelos
        return models;
    }

    private static SMO evaluateKernel(Instances train, weka.classifiers.functions.supportVector.Kernel kernel, double kernelParam) throws Exception {
        double[] cValues = {0.1, 1, 10}; // Valores de C a probar
        double[] gammaValues = {0.001, 0.01, 0.1, 1}; // Rango más razonable
        double[] exponentValues = {1, 2, 3}; // Valores de Exponent (para PolyKernel)
        double[] omegaValues = {0.5, 1.0, 2.0}; // Valores de Omega (para PukKernel)
        double[] sigmaValues = {0.5, 1.0, 2.0}; // Valores de Sigma (para PukKernel)

        double bestWeightedFMeasure = 0; // Mejor weighted average F-measure
        double bestC = 0;
        double bestKernelParam1 = 0; // Puede ser Gamma, Exponent, Omega, etc.
        double bestKernelParam2 = 0; // Solo para PukKernel (Sigma)
        SMO bestModel = null;

        for (double c : cValues) {
            for (double param1 : (kernel instanceof weka.classifiers.functions.supportVector.RBFKernel ? gammaValues :
                                  kernel instanceof weka.classifiers.functions.supportVector.PolyKernel ? exponentValues :
                                  kernel instanceof weka.classifiers.functions.supportVector.Puk ? omegaValues : new double[]{kernelParam})) {
                for (double param2 : (kernel instanceof weka.classifiers.functions.supportVector.Puk ? sigmaValues : new double[]{1.0})) {
                    // Crear el clasificador SMO
                    SMO smo = new SMO();
                    smo.setKernel(kernel);
                    smo.setC(c);

                    // Configurar parámetros específicos del kernel
                    if (kernel instanceof weka.classifiers.functions.supportVector.PolyKernel) {
                        ((weka.classifiers.functions.supportVector.PolyKernel) kernel).setExponent((int) param1);
                    } else if (kernel instanceof weka.classifiers.functions.supportVector.RBFKernel) {
                        ((weka.classifiers.functions.supportVector.RBFKernel) kernel).setGamma(param1);
                    } else if (kernel instanceof weka.classifiers.functions.supportVector.Puk) {
                        ((weka.classifiers.functions.supportVector.Puk) kernel).setOmega(param1);
                        ((weka.classifiers.functions.supportVector.Puk) kernel).setSigma(param2);
                    }

                    // Evaluar el modelo con validación cruzada
                    Evaluation eval = new Evaluation(train);
                    eval.crossValidateModel(smo, train, 10, new Random(1)); // 10-fold cross-validation

                    // Obtener el weighted average F-measure
                    double weightedFMeasure = eval.weightedFMeasure();

                    // Imprimir resultados
                    System.out.println("Kernel: " + kernel.getClass().getSimpleName());
                    System.out.println("C: " + c);
                    System.out.println("Param1: " + param1 + (kernel instanceof weka.classifiers.functions.supportVector.Puk ? ", Param2: " + param2 : ""));
                    System.out.println("Weighted F-Measure: " + weightedFMeasure);
                    System.out.println("====================================");

                    // Actualizar el mejor modelo si el weighted F-measure mejora
                    if (weightedFMeasure > bestWeightedFMeasure) {
                        bestWeightedFMeasure = weightedFMeasure;
                        bestC = c;
                        bestKernelParam1 = param1;
                        bestKernelParam2 = param2;
                        bestModel = smo;
                    }
                }
            }
        }

        // Entrenar el mejor modelo con el conjunto de entrenamiento completo
        if (bestModel != null) {
            bestModel.buildClassifier(train);
        }

        // Imprimir el mejor valor de parámetros
        System.out.println("Best C for " + kernel.getClass().getSimpleName() + ": " + bestC);
        System.out.println("Best Param1: " + bestKernelParam1 + (kernel instanceof weka.classifiers.functions.supportVector.Puk ? ", Best Param2: " + bestKernelParam2 : ""));
        System.out.println("Best Weighted F-Measure: " + bestWeightedFMeasure);
        System.out.println("====================================");

        // Retornar el mejor modelo con los mejores parámetros
        return bestModel;
    }

}