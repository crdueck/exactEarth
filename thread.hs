import Control.Monad
import Control.Monad.Trans
import Data.Sequence -- Queue with O(1) head and tail operations

data Thread m r
    = Yield   (Thread m r)
    | Fork    (Thread m r) (Thread m r)
    | Lift (m (Thread m r))
    | Pure  r
    | Done

instance Monad m => Functor (Thread m) where
    fmap f = go where
        go (Yield  t) = Yield (go t)
        go (Fork l r) = Fork  (go l) (go r)
        go (Lift   m) = Lift  (liftM go m)
        go (Pure   r) = Pure  (f r)
        go  Done      = Done

instance Monad m => Monad (Thread m) where
    return   = Pure
    t0 >>= f = go t0 where
        go (Yield  t) = Yield (go t)
        go (Fork l r) = Fork  (go l) (go r)
        go (Lift   m) = Lift  (liftM go m)
        go (Pure   r) = f r
        go  Done      = Done

instance MonadTrans Thread where
    lift = Lift . liftM Pure

yield :: Monad m => Thread m ()
yield = Yield (Pure ())

done :: Monad m => Thread m r
done = Done

fork :: Monad m => Thread m r -> Thread m ()
fork thread = do
    child <- Fork (Pure False) (Pure True)
    when child $ thread >> done

roundRobin :: Monad m => Thread m a -> m ()
roundRobin = go . singleton
  where
    go ts = case viewl ts of
        -- The queue is empty: we're done!
        EmptyL   -> return ()
        -- The queue is non-empty: Process the first thread
        t :< ts -> case t of
            -- Yielding threads go to the back of the queue
            Yield   t' -> go (ts |> t')
            -- New threads go to the back of the queue
            Fork t1 t2 -> go (t1 <| (ts |> t2))
            -- Run effects in the base monad
            Lift    m  -> m >>= \t' -> go (t' <| ts)
            -- Thread done: Remove the thread from the queue
            _          -> go ts

mainThread :: Thread IO ()
mainThread = do
    lift $ putStrLn "Forking thread #1"
    fork thread1
    lift $ putStrLn "Forking thread #2"
    fork thread2

thread1 :: Thread IO ()
thread1 = forM_ [1..10] $ \i -> do
    lift $ print i
    yield

thread2 :: Thread IO ()
thread2 = replicateM_ 3 $ do
    lift $ putStrLn "Hello World"
    yield

main = roundRobin mainThread
