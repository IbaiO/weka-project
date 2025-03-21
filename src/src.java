package src;

import java.io.File;
import java.io.FileReader;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class src {
    public static void main(String[] args) {
        ///////////// HASIERAKETAK /////////////
        Instances instances = null;

        ///////////// IRAKURKETA /////////////
        try {
            FileReader fi = new FileReader(args[1]);
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
            saver.setFile(new File(args[2]));
            saver.writeBatch();
            System.out.println("Datuak gorde dira: " + args[2]);
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da " + args[2] + " fitxategia gorde.");
            e.printStackTrace();
        }
    }
}
