package src;
import java.io.File;
import java.io.FileReader;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

@SuppressWarnings("all")
public class main {
    public static void main(String[] args) throws Exception {
        ///////////// HASIERAKETAK /////////////
        if (args.length != 2) {
            System.out.println("Erabilera: java -jar weka-project.jar <input karpetaren path-a> <output file-a (extentzio gabe)>");
            System.exit(0);
        }
        File dir = new File(args[0]);
        if (!dir.isDirectory() || args[1].contains(".")) {
            System.out.println("Erabilera: java -jar weka-project.jar <input karpetaren path-a> <output file-a (extentzio gabe)>");
            System.exit(0);
        }
        String inputPath = args[0]; System.out.println("Sarrerako fitxategia: " + inputPath);
        String outputFile = args[1]; System.out.println("Irteerako fitxategia: " + outputFile);
        Instances instances[] = null;
        
        // Configurar netlib-java para usar implementaciones en Java puro
        System.setProperty("com.github.fommil.netlib.BLAS", "com.github.fommil.netlib.F2jBLAS");
        System.setProperty("com.github.fommil.netlib.LAPACK", "com.github.fommil.netlib.F2jLAPACK");

        ///////////// PROCESAMENDUA /////////////
        instances = datuBilketa.getDB().bildu(inputPath, outputFile);
        iragarri.main(instances[0], instances[1]);
        // bostgarren laborategia
        // SMO algoritmoa
        // Linear regression algoritmoa   
        
        
        System.exit(0);


        ///////////// IDAZKETA /////////////
        ArffSaver saver = new ArffSaver();
        saver.setInstances(instances[0]);
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
