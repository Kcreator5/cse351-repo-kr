using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

namespace assignment11;

public class Assignment11
{
    private const long START_NUMBER = 10_000_000_000;
    private const int RANGE_COUNT = 1_000_000;
    private const int WORKER_COUNT = 4;

    private static int numbersProcessed = 0;
    private static int primeCount = 0;
    private static bool IsPrime(long n)
    {
        if (n <= 3) return n > 1;
        if (n % 2 == 0 || n % 3 == 0) return false;

        for (long i = 5; i * i <= n; i += 6)
        {
            if (n % i == 0 || n % (i + 2) == 0)
                return false;
        }
        return true;
    }

    public static void Main(string[] args)
    {
        var queue = new BlockingCollection<long>(boundedCapacity: 1000);

        var stopwatch = Stopwatch.StartNew();

        // 🔹 Start worker threads
        Task[] workers = new Task[WORKER_COUNT];

        for (int i = 0; i < WORKER_COUNT; i++)
        {
            workers[i] = Task.Run(() =>
            {
                foreach (var number in queue.GetConsumingEnumerable())
                {
                    Interlocked.Increment(ref numbersProcessed);

                    if (IsPrime(number))
                    {
                        Interlocked.Increment(ref primeCount);
                        Console.Write($"{number}, ");
                    }
                }
            });
        }

        // 🔹 main thread for adding numbers for the workers to check later
        for (long i = START_NUMBER; i < START_NUMBER + RANGE_COUNT; i++)
        {
            queue.Add(i);
        }

        // Signals no more stuff
        queue.CompleteAdding();

        // this waits for workers to finish
        Task.WaitAll(workers);

        stopwatch.Stop();

        Console.WriteLine("\n");
        Console.WriteLine($"Numbers processed = {numbersProcessed}");
        Console.WriteLine($"Primes found      = {primeCount}");
        Console.WriteLine($"Total time        = {stopwatch.Elapsed}");
    }
}

/*
using System.Diagnostics;

namespace assignment11;

public class Assignment11
{
    private const long START_NUMBER = 10_000_000_000;
    private const int RANGE_COUNT = 1_000_000;

    private static bool IsPrime(long n)
    {
        if (n <= 3) return n > 1;
        if (n % 2 == 0 || n % 3 == 0) return false;

        for (long i = 5; i * i <= n; i = i + 6)
        {
            if (n % i == 0 || n % (i + 2) == 0)
                return false;
        }
        return true;
    }

    public static void Main(string[] args)
    {
        // Use local variables for counting since we are in a single thread.
        int numbersProcessed = 0;
        int primeCount = 0;

        Console.WriteLine("Prime numbers found:");

        var stopwatch = Stopwatch.StartNew();
        
        // A single for-loop to check every number sequentially.
        for (long i = START_NUMBER; i < START_NUMBER + RANGE_COUNT; i++)
        {
            numbersProcessed++;
            if (IsPrime(i))
            {
                primeCount++;
                Console.Write($"{i}, ");
            }
        }

        stopwatch.Stop();

        Console.WriteLine(); // New line after all primes are printed
        Console.WriteLine();

        // Should find 43427 primes for range_count = 1000000
        Console.WriteLine($"Numbers processed = {numbersProcessed}");
        Console.WriteLine($"Primes found      = {primeCount}");
        Console.WriteLine($"Total time        = {stopwatch.Elapsed}");        
    }
}
*/