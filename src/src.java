package src;

import java.io.File;
import java.io.FileReader;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class src {
    public static void main(String[] args) {
        ///////////// HASIERAKETAK /////////////
        if (args.length != 2) {
            System.out.println("Erabilera: java -jar src.jar <input.arff> <output.arff>");
            System.exit(0);
        }
        String inputFile = args[0]; System.out.println("Sarrerako fitxategia: " + inputFile);
        String outputFile = args[1]; System.out.println("Irteerako fitxategia: " + outputFile);
        Instances instances = null;

        ///////////// IRAKURKETA /////////////
        try {
            FileReader fi = new FileReader(args[0]);
            instances = new Instances(fi);
            fi.close();
            instances.setClassIndex(instances.numAttributes() - 1);
        } catch (Exception e) {
            System.out.println("Errorea: " + e.getMessage());
        }
        
        ///////////// PROCESAMENDUA /////////////
        // datuak garbitzea
        // seigarrel laborategia
        // bostgarren laborategia
        // VSM algoritmoa
        // Linear regression algoritmoa        

        ///////////// IDAZKETA /////////////
        ArffSaver saver = new ArffSaver();
        saver.setInstances(instances);
        try {
            saver.setFile(new File(outputFile));
            saver.writeBatch();
            System.out.println("Datuak gorde dira: " + outputFile);
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da " + outputFile + " fitxategia gorde.");
            e.printStackTrace();
        }
    }
}
