// zsthampi - Zubin Thampi

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

import java.io.IOException;
import java.util.*;

/*
 * Main class of the TFIDF MapReduce implementation.
 * Author: Tyler Stocksdale
 * Date:   10/18/2017
 */
public class TFIDF {

    public static void main(String[] args) throws Exception {
        // Check for correct usage
        if (args.length != 1) {
            System.err.println("Usage: TFIDF <input dir>");
            System.exit(1);
        }
		
		// Create configuration
		Configuration conf = new Configuration();
		
		// Input and output paths for each job
		Path inputPath = new Path(args[0]);
		Path wcInputPath = inputPath;
		Path wcOutputPath = new Path("output/WordCount");
		Path dsInputPath = wcOutputPath;
		Path dsOutputPath = new Path("output/DocSize");
		Path tfidfInputPath = dsOutputPath;
		Path tfidfOutputPath = new Path("output/TFIDF");
		
		// Get/set the number of documents (to be used in the TFIDF MapReduce job)
        FileSystem fs = inputPath.getFileSystem(conf);
        FileStatus[] stat = fs.listStatus(inputPath);
		String numDocs = String.valueOf(stat.length);
		conf.set("numDocs", numDocs);
		
		// Delete output paths if they exist
		FileSystem hdfs = FileSystem.get(conf);
		if (hdfs.exists(wcOutputPath))
			hdfs.delete(wcOutputPath, true);
		if (hdfs.exists(dsOutputPath))
			hdfs.delete(dsOutputPath, true);
		if (hdfs.exists(tfidfOutputPath))
			hdfs.delete(tfidfOutputPath, true);
		
		// Create and execute Word Count job
		Job wcJob = Job.getInstance(conf, "word count");
		wcJob.setMapperClass(WCMapper.class);
		wcJob.setReducerClass(WCReducer.class);
		wcJob.setOutputKeyClass(Text.class);
		wcJob.setOutputValueClass(IntWritable.class);
		FileInputFormat.addInputPath(wcJob, wcInputPath);
		FileOutputFormat.setOutputPath(wcJob, wcOutputPath);
		wcJob.waitForCompletion(true);
			
		// Create and execute Document Size job
		Job dsJob = Job.getInstance(conf, "doc size");
		dsJob.setMapperClass(DSMapper.class);
		dsJob.setReducerClass(DSReducer.class);
		dsJob.setOutputKeyClass(Text.class);
		dsJob.setOutputValueClass(Text.class);
		FileInputFormat.addInputPath(dsJob, dsInputPath);
		FileOutputFormat.setOutputPath(dsJob, dsOutputPath);
		dsJob.waitForCompletion(true);
		
		//Create and execute TFIDF job
		Job tfidfJob = Job.getInstance(conf, "tfidf");
		tfidfJob.setMapperClass(TFIDFMapper.class);
		tfidfJob.setReducerClass(TFIDFReducer.class);
		tfidfJob.setOutputKeyClass(Text.class);
		tfidfJob.setOutputValueClass(Text.class);
		FileInputFormat.addInputPath(tfidfJob, tfidfInputPath);
		FileOutputFormat.setOutputPath(tfidfJob, tfidfOutputPath);
		tfidfJob.waitForCompletion(true);
    }
	
	/*
	 * Creates a (key,value) pair for every word in the document 
	 *
	 * Input:  ( byte offset , contents of one line )
	 * Output: ( (word@document) , 1 )
	 *
	 * word = an individual word in the document
	 * document = the filename of the document
	 */
	public static class WCMapper extends Mapper<Object, Text, Text, IntWritable> {
		
		private final static IntWritable one = new IntWritable(1);
		private Text word = new Text();

		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			StringTokenizer itr = new StringTokenizer(value.toString());
			while (itr.hasMoreTokens()) {
				word.set(itr.nextToken().concat("@").concat(((FileSplit) context.getInputSplit()).getPath().getName()));
				context.write(word, one);
			}
		}
		
    }

    /*
	 * For each identical key (word@document), reduces the values (1) into a sum (wordCount)
	 *
	 * Input:  ( (word@document) , 1 )
	 * Output: ( (word@document) , wordCount )
	 *
	 * wordCount = number of times word appears in document
	 */
	public static class WCReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
		
		private IntWritable result = new IntWritable();

		public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
			int sum = 0;
			for (IntWritable val: values) {
				sum += val.get();
			}
			result.set(sum);
			context.write(key, result);
		}
    }
	
	/*
	 * Rearranges the (key,value) pairs to have only the document as the key
	 *
	 * Input:  ( (word@document) , wordCount )
	 * Output: ( document , (word=wordCount) )
	 */
	public static class DSMapper extends Mapper<Object, Text, Text, Text> {
		
		private Text document = new Text();
		private Text result = new Text();

		public void map(Object key, Text line, Context context) throws IOException, InterruptedException {
			String s = line.toString();
			document.set(s.split("\t")[0].trim().split("@")[1]);
			result.set(s.split("\t")[0].trim().split("@")[0].concat("=").concat(s.split("\t")[1].trim()));
			context.write(document, result);
		}
		
    }

    /*
	 * For each identical key (document), reduces the values (word=wordCount) into a sum (docSize) 
	 *
	 * Input:  ( document , (word=wordCount) )
	 * Output: ( (word@document) , (wordCount/docSize) )
	 *
	 * docSize = total number of words in the document
	 */
	public static class DSReducer extends Reducer<Text, Text, Text, Text> {
		
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			int sum = 0;
			int size = 0;

			// Keep a cache of elements, so we can iterate again
			ArrayList<Text> cache = new ArrayList<Text>();

			for (Text val: values) {
				Text tmp = new Text();
				tmp.set(val);
				sum += Integer.parseInt(val.toString().split("=")[1]);
				size += 1;
				cache.add(tmp);
			}
			
			for (int i=0; i<size; i++) {
				Text word = new Text();
				Text result = new Text(); 
				Text val = new Text();
				val = cache.get(i);
				word.set(val.toString().split("=")[0].trim().concat("@").concat(key.toString()));
				result.set(val.toString().split("=")[1].trim().concat("/").concat(Integer.toString(sum)));
				context.write(word, result);
			}
		}
    }
	
	/*
	 * Rearranges the (key,value) pairs to have only the word as the key
	 * 
	 * Input:  ( (word@document) , (wordCount/docSize) )
	 * Output: ( word , (document=wordCount/docSize) )
	 */
	public static class TFIDFMapper extends Mapper<Object, Text, Text, Text> {

		public void map(Object key, Text line, Context context) throws IOException, InterruptedException {
			String s = line.toString();
			Text first = new Text();
			Text second = new Text(); 
			first.set(s.split("\t")[0].trim().split("@")[0].trim());
			second.set(s.split("\t")[0].trim().split("@")[1].trim().concat("=").concat(s.split("\t")[1].trim()));
			context.write(first, second);
		}
		
    }

    /*
	 * For each identical key (word), reduces the values (document=wordCount/docSize) into a 
	 * the final TFIDF value (TFIDF). Along the way, calculates the total number of documents and 
	 * the number of documents that contain the word.
	 * 
	 * Input:  ( word , (document=wordCount/docSize) )
	 * Output: ( (document@word) , TFIDF )
	 *
	 * numDocs = total number of documents
	 * numDocsWithWord = number of documents containing word
	 * TFIDF = (wordCount/docSize) * ln(numDocs/numDocsWithWord)
	 *
	 * Note: The output (key,value) pairs are sorted using TreeMap ONLY for grading purposes. For
	 *       extremely large datasets, having a for loop iterate through all the (key,value) pairs 
	 *       is highly inefficient!
	 */
	public static class TFIDFReducer extends Reducer<Text, Text, Text, Text> {
		
		private static int numDocs;
		private Map<Text, Text> tfidfMap = new HashMap<>();
		
		// gets the numDocs value and stores it
		protected void setup(Context context) throws IOException, InterruptedException {
			Configuration conf = context.getConfiguration();
			numDocs = Integer.parseInt(conf.get("numDocs"));
		}
		
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			// Keep a cache of elements, so we can iterate again
			ArrayList<Text> cache = new ArrayList<Text>();	
			int numDocsWithWord = 0;
			
			for (Text val: values) {
				Text tmp = new Text();
				tmp.set(val);
				cache.add(tmp);
				numDocsWithWord += 1;
			}

	 		for (int i=0; i<numDocsWithWord; i++) {
	 			Text word = new Text();
	 			Text tfidf = new Text();
	 			Text val = new Text();

	 			val = cache.get(i);
	 			word.set(val.toString().split("=")[0].trim().concat("@").concat(key.toString()));
	 			
	 			double wordCount = Double.parseDouble(val.toString().split("=")[1].trim().split("/")[0].trim());
	 			double docSize = Double.parseDouble(val.toString().split("=")[1].trim().split("/")[1].trim());
	 			tfidf.set(Double.toString((wordCount/docSize) * (Math.log((double) numDocs/(double) numDocsWithWord))));
	 			
	 			//Put the output (key,value) pair into the tfidfMap instead of doing a context.write
				tfidfMap.put(word, tfidf);	
	 		}
		}
		
		// sorts the output (key,value) pairs that are contained in the tfidfMap
		protected void cleanup(Context context) throws IOException, InterruptedException {
            Map<Text, Text> sortedMap = new TreeMap<Text, Text>(tfidfMap);
			for (Text key : sortedMap.keySet()) {
                context.write(key, sortedMap.get(key));
            }
        }
		
    }
}
