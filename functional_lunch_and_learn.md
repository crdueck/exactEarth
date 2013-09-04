# Haskell at a Glance

~~~ haskell
data List a = [] | a : List a
-- []  :: List a
-- (:) :: a -> List a -> List a

myList :: [Int]
myList = 1 : (2 : (3 : []))
-- or myList = [1, 2, 3]
-- or myList = [1..3]

-- pattern matching
null :: [a] -> Bool
null []     = True
null (x:xs) = False

-- curried function
add3 :: Int -> Int -> Int -> Int
add3 x y z = x + y + z

-- partially applied function
add2 :: Int -> Int -> Int
add2 = add3 0

-- wildcards
const :: a -> b -> a
const x _ = x
~~~

# What is Functional Programming?

Functional programming is a style of programming that promotes the use of

* pure functions: "mathematical" functions that depend only on their inputs, not
the global program state.

* immutable data structures: once defined, data cannot be mutated. Instead,
a new modified copy of the data is created.

* declarative style: describe "what" the solution to a problem is, not "how" to
do it.

* implicit recursion: recursion is encouraged over iteration, but higher order
functions abstract explicit recursion away.

# Why Use Functional Programming?

Correct use of a functional style results in code that is easier to maintain and
reason about.

Functional code is often "obviously" correct and very composable (read:
reusable) due to the use of higher order functions (functions that take
functions as arguments to change their behaviour).

Pure functions and immutable data structures reduce the cognitive load required
to understand a piece of code. The behaviour of a function will never depend on
a global variable defined halfway across the code base, or the current state of
the program. Everything a programmer needs to reason about a piece of code is
self contained.

All the code we will ever write will eventually be executed on some inherently
procedural CPU. However, our code will also have to be modified many times over
by other programmers to add new features, fix bugs and optimize for performance.

Functional programming lets us write clear, correct code that is easy to
maintain, while letting powerful optimizing compilers do the "busy work" of
translating our high level code for humans to low level code for machines.

# Essential Higher Order Functions

There are some very frequently used higher order functions that capture common
recursive operations over `List`. We'll see later that most of these functions
can be generalized to work on other data structures besides `List`.

## Map

~~~ haskell
map :: (a -> b) -> [a] -> [b]
map _ []     = []
map f (x:xs) = f x : map f xs
~~~

Let's apply `map` step by step to see how it works.

~~~ haskell
map odd [1,2,3] = odd 1 : map odd [2,3]
                = odd 1 : odd 2 : map odd [3]
                = odd 1 : odd 2 : odd 3 : map odd []
                = odd 1 : odd 2 : odd 3 : []
                = [True, False, True]
~~~

Basically, `map` replaces each element in a list with the result of applying the
function to that element.

## Map

Suppose we have a `List` of input data. We want to process each item in some way
and store the results in a new `List`. We might do this using the following
imperative code.

~~~ scala
def processList(input: List[Input]) : List[Output] = {
    var out = new List[Output]()

    for (data <- input) {
        out += processData(data)
    }
    return out
}
~~~

Now a functional version using `map`

~~~ scala
def processList(input: List[Input]) : List[Output] = {
   input.map(processData)
}
~~~

Using `map` improves code readability. It may be hard to tell what a handwritten
for loop does at a glance, but `map`:

* always traverses each element
* always applies the same function to each element in the same way

# Filter

~~~ haskell
filter :: (a -> Bool) -> [a] -> [a]
filter _ []     = []
filter p (x:xs) =
    if p x
       then x : filter p xs
       else     filter p xs
~~~

Let apply `filter` step by step to see how it works.

~~~ haskell
filter odd [1,2,3] = 1 : filter odd [2,3]  -- 1 is odd
                   = 1 : filter odd [3]    -- 2 is not odd
                   = 1 : 3 : filter odd [] -- 3 is odd
                   = 1 : 3 : []
                   = [1, 3]
~~~

An iterative function to remove odd numbers from a `List`

~~~ scala
def onlyEvens(xs: List[Int]) : List[Int] = {
    var acc = new List[Int]()
    for (x <- xs) {
        if (x % 2 == 0) {
            acc += x
        }
    }
    return acc
}
~~~

and a functional version using `filter`

~~~ scala
def onlyEvens(xs: List[Int]) : List[Int] = {
    xs.filter(_ % 2 == 0)
}
~~~

# Foldr

~~~ haskell
foldr :: (a -> b -> b) -> b -> [a] -> b
foldr _ acc []     = acc
foldr f acc (x:xs) = f x (foldr f acc xs)
~~~

Lets apply `foldr` step by step to see how it works.

~~~ haskell
sum :: [Int] -> Int
sum xs = foldr (+) 0 xs
~~~

~~~ haskell
sum [1,2,3] = foldr (+) 0 [1,2,3]
            = 1 + (foldr (+) 0 [2,3])
            = 1 + (2 + (foldr (+) 0 [3]))
            = 1 + (2 + (3 + (foldr (+) 0 [])))
            = 1 + (2 + (3 + 0))
            = 1 + (2 + 3)
            = 1 + 5
            = 6
~~~

Like `map`, `fold` has strong properties that make it easy to reason about. Of
them, a `fold`:

* always traverses each element in the collection you're folding over

* always applies the same combining function to each element in the same way

`foldr` has an important *Universal Property* that allows compilers to perform
an optimization called `foldr/build` fusion to reduce the amount of allocations
required in code that processes lists heavily.

# Foldr

Here's an iterative factorial function

~~~ scala
def factorial(n: Int): Int = {
    var res = 1
    for (x <- 1 until n + 1) {
        res *= x
    }
    return res
}
~~~

and the functional version using `foldr`

~~~ scala
def factorial(n: Int): Int = {
    (1 until n + 1).fold(1)(_*_)
}
~~~

# Foldr

Many common functions are `foldr` in disguise!

~~~ haskell
reverse :: [a] -> [a]
reverse xs = foldr (\x xs -> xs ++ [x]) [] xs
-- reverse = foldl (flip (:)) []
-- foldl :: (a -> b -> a) -> a -> [b] -> a

maximum :: [Int] -> Int
maximum (x:xs) = foldr max x xs

concat :: [[a]] -> [a]
concat xs = foldr (++) [] xs

all :: [Bool] -> Bool
all xs = foldr (&&) True xs

any :: [Bool] -> Bool
any xs = foldr (||) False xs

filter :: (a -> Bool) -> [a] -> [a]
filter p xs = foldr (\x xs -> if p x then x : xs else xs) [] xs

map :: a -> b -> [a] -> [b]
map f xs = foldr (\x xs -> f x : xs) [] xs
~~~

# Foldr

What do all of these functions have in common?  They all take a `List` of
"things" and combine them together in some way to give you one "thing" back
(possibly of a different type).

~~~ haskell
foldr :: (a -> b -> b) -- combining function
      ->  b            -- initial value
      -> [a]           -- list of a's
      ->  b            -- final value
~~~

Think of folds whenever you want to condense a `List` of "things" into a single
"thing".

But `List` aren't the only things we can fold.

~~~ haskell
data Tree a = Leaf a | Branch (Tree a) a (Tree a)

foldTree :: (a -> b -> b) -> b -> Tree a -> b
foldTree f z (Leaf a)       = f a z
foldTree f z (Branch l a r) = f a (foldTree f (foldTree f z r) l)
~~~

# Foldr

In fact, many data structures support folding operations. Haskell defines
a `Foldable` typeclass.

~~~ haskell
class Foldable f where
    foldr :: (a -> b -> b) -> b -> f a -> b
    foldl :: (a -> b -> a) -> a -> f b -> a
    foldMap :: Monoid m => (a -> m) -> m -> f a -> m
    ...
~~~

For example, `Maybe` has a `Foldable` instance.

~~~ haskell
instance Foldable Maybe where
    foldr _ z Nothing  = z
    foldr f z (Just a) = f a z
~~~

You can always expect a `fold` over any foldable structure to behave like
a `fold` over a `List`. We can write this as a "law" that says

~~~ haskell
Foldable.foldr f z xs == List.foldr f z (toList xs)
~~~

Scala doesnt directly define a `Foldable` class or trait, instead classes like
`List`, `Map` and `Option` provide their own `foldRight` and `foldLeft` methods
that behave just like `foldr` and `foldl`.

# Foldr

Folds also encompass a simple sort of state, as the combining function is
applied to the result of the previous application.

~~~ haskell
foldr :: (Input -> Result -> Result) -> Result -> [Input] -> Result
~~~

Each time the function is called, it has access to the intermediate result as it
has been computed up to that point. This intermediate result is used to compute
the next intermediate result and so on.

So, think of folds whenever you want to repeatedly apply a function to a
collection of inputs and the function depends the result of previous
applications.

# More Recursive Primitives

There are many more basic recursive functions in a functional programmers
arsenal. It would be time consuming to explain them all, but here are a few
examples.

~~~ haskell
zipWith :: (a -> b -> c) -> [a] -> [b] -> [c]

zip :: [a] -> [b] -> [(a, b)]
-- zip == zipWith (\a -> b -> (a, b))

unfoldr :: (b -> Maybe (a, b)) -> b -> [a]

iterate :: (a -> a) -> a -> [a]

find :: (a -> Bool) -> [a] -> Maybe a
~~~

# Intermediate Functional Concepts

## Higher Order Types

First we need to understand higher order (or partially applied) types. A good
analogy are generics in langauges like Java and C++. Higher order types are
applied to concrete types to create new concrete types.

Haskell examples

~~~ haskell
Int    :: *
[]     :: * -> *
[Int]  :: *
(->)   :: * -> * -> *
a -> b :: *
~~~

Scala examples

~~~ scala
Int       :: *
List      :: * -> *
List[Int] :: *
Option    :: * -> *
Option[List[Int]] :: *
Option[List] // kind mis-match!
~~~

The "type" of a type is called a kind. We say `Option` has kind `* -> *`.

# Functors

~~~ haskell
class Functor (f :: * -> *) where
    fmap :: (a -> b) -> (f a -> f b)
~~~

A `Functor` is formally defined in Category Theory. Don't worry, you dont need
to understand any of this to use `Functor`!

Let $C$ and $D$ be categories. A `Functor` from $C$ to $D$ is a mapping that

* associates each object ${X \in C}$ with an object ${F(X) \in D}$

* associates each morphism ${\phi : X \rightarrow Y \in C}$ with a morphism

    ${F(\phi) : F(X) \rightarrow F(Y) \in D}$

    such that the following two conditions hold:

    - ${F(id_x) = id_{F(X)}}$ for all objects ${X \in C}$
    - ${F(\phi \circ \psi) = F(\phi) \circ F(\psi)}$ for all morphisms ${\phi : X \rightarrow Y, \psi : Y \rightarrow Z}$

We can write these laws in Haskell as

~~~ haskell
fmap id == id
fmap (f . g) == fmap f . fmap g
~~~

# Functors

This is a very generalized mathematical definition. In practice a programming
language is only concerned with one category: the category of types
representable in the language (`Hask` in Haskell).

In this category, objects are just the types of kind `*`

~~~ haskell
Int, Bool, Maybe Char, a
~~~

and morphisms are just functions.

~~~ haskell
Int -> Bool, [a] -> Int, a -> b
~~~

So our requirements for a `Functor` becomes

* associates each type `a` with a type `f a`

~~~ haskell
a -> f a
~~~

* associates each function `a -> b` with a function `f a -> f b` 

~~~ haskell
(a -> b) -> (f a -> f b)
~~~

# Functors

Let take another look at the `Functor` typeclass

~~~ haskell
class Functor (f :: * -> *) where
    fmap :: (a -> b) -> (f a -> f b)
~~~

The definition for the `Functor` class only defines the second requirement!

It seems like something is missing, until we realize the constructor for our `Functor`
satisfies the first mapping, and so all is well.

# Functors

Here are some examples to help this sink in

~~~ haskell
newtype Identity a = Id a

instance Functor Identity where
    fmap :: (a -> b) -> (Identity a -> Identity b)
    fmap f (Id a) = Id (f a)

instance Functor [] where
    -- looks very similar to map!
    fmap :: (a -> b) -> ([a] -> [b])
    fmap _ []     = []
    fmap f (x:xs) = f x : fmap f xs

data Maybe a = Nothing | Just a

instance Functor Maybe where
    fmap :: (a -> b) -> (Maybe a -> Maybe b)
    fmap _ Nothing  = Nothing
    fmap f (Just a) = Just (f a)

data Tree a = Leaf a | Branch (Tree a) a (Tree a)

instance Functor Tree where
    fmap :: (a -> b) -> (Tree a -> Tree b)
    fmap f (Leaf a)     = Leaf (f a)
    fmap f (Branch l r) = Branch (fmap f l) (f a) (fmap f r)
~~~

In a language with `Future`, we can make `Future` an instance of `Functor`. How
would `fmap` behave?

# Functors

~~~ haskell
data Thread r
    = Yield (Thread r)
    | Fork  (Thread r) (Thread r)
    | Pure  r
~~~

~~~ haskell
instance Functor Thread where
    fmap f (Yield next)      = ...
~~~

# Functors

~~~ haskell
data Thread r
    = Yield (Thread r)
    | Fork  (Thread r) (Thread r)
    | Pure  r
~~~

~~~ haskell
instance Functor Thread where
    fmap f (Yield next)      = Yield (fmap f next)
    fmap f (Fork left right) = ...
~~~

# Functors

~~~ haskell
data Thread r
    = Yield (Thread r)
    | Fork  (Thread r) (Thread r)
    | Pure  r
~~~

~~~ haskell
instance Functor Thread where
    fmap f (Yield next)      = Yield (fmap f next)
    fmap f (Fork left right) = Fork (fmap f left) (fmap f right)
    fmap f (Pure r)          = ... 
~~~

# Functors

~~~ haskell
data Thread r
    = Yield (Thread r)
    | Fork  (Thread r) (Thread r)
    | Pure  r
~~~

~~~ haskell
instance Functor Thread where
    fmap f (Yield next)      = Yield (fmap f next)
    fmap f (Fork left right) = Fork (fmap f left) (fmap f right)
    fmap f (Pure r)          = Pure (f r)
~~~

# Functors

We can think of a `Functor` as some kind of "context" around a pure value.  We
use `fmap` to lift a function on pure values to a function on values in this
"context", and `fmap` automagically knows what instance of `Functor` to use!

Scala by default doesn't define `Functor`, instead classes like `List`, `Map`
and `Option` define their own `map` methods that "do the right thing". These are
exactly the same as their would-be `Functor` instances.

# Monads

> "A monad is just a monoid in the category of endofunctors, what's the problem?"

`Monads` may seem complicated at first, but are really very simple

~~~ haskell
class Functor m => Monad (m :: * -> *) where
    return :: a -> m a

    join :: m (m a) -> m a
    join mm = mm >>= id

    (>>=) :: m a -> (a -> m b) -> m b
    m >>= f = join (fmap f m)
~~~

We can see from the definition that a `Monad` must also be a `Functor` Indeed,
a `Monad` is strictly more powerful than a `Functor` because we can derive
a `Functor` instance from any `Monad`.

~~~ haskell
fmap :: Monad m => (a -> b) -> (m a -> m b)
fmap f m = m >>= \a -> return (f a)
~~~

# Monads

`Monads` are also formally defined in category theory:

Let $C$ be a category. A `Monad` on $C$ consists of a (endo)functor
$T : C \rightarrow C$, together with two natural transformations

$unit : X \rightarrow T(X)$ and $join : T(T(X)) \rightarrow T(X)$

such that the following two conditions hold:

* $join \circ T(join) = join \circ join$
* $join \circ unit = id = join \circ T(unit)$

The formal definition uses `join`, but the more useful operation in practice is
`(>>=)`, pronounced "bind". In terms of `(>>=)` these laws become

Left Identity

~~~ haskell
return a >>= f == f a
~~~

Right Identity

~~~ haskell
m >>= return == m
~~~

Associativity

~~~ haskell
(m >>= f) >>= g == m >>= (\x -> f x >>= g)
-- (m >>= f) >>= g == m >>= (f >=> g)
~~~

# Monads

To make a `Monad` instance, we need two things

* a function to create a monadic action that does nothing but return a value
* a function to apply a monadic function to the result of a monadic action

Some example `Monad` instances

~~~ haskell
instance Monad Identity where
    return a = Id a
    join (Id (Id a)) = Id a

    (>>=) :: Identity a -> (a -> Identity b) -> Identity b
    Id a >>= f = f a

instance Monad List where
    return a = [a]
    join xs = concat xs

    (>>=) :: [a] -> (a -> [b]) -> [b]
    xs >>= f = concat (map f xs)

instance Monad Maybe where
    return a = Just a

    (>>=) :: Maybe a -> (a -> Maybe b) -> Maybe b
    Nothing >>= _ = Nothing
    Just a  >>= f = f a

instance Monad Tree where
    return a = Leaf a

    (>>=) :: Tree a -> (a -> Tree b) -> Tree b
    Leaf a >>= f = f a
    Branch l r >>= f = Branch (l >>= f) (r >>= f)
~~~

# Monads

~~~ haskell
data Thread r
    = Yield (Thread r)
    | Fork  (Thread r) (Thread r)
    | Pure  r

instance Monad Thread where
    return r = ...
~~~

# Monads

~~~ haskell
data Thread r
    = Yield (Thread r)
    | Fork  (Thread r) (Thread r)
    | Pure  r

instance Monad Thread where
    return r = Pure r
    
    Yield next    >>= f = ...
~~~

# Monads

~~~ haskell
data Thread r
    = Yield (Thread r)
    | Fork  (Thread r) (Thread r)
    | Pure  r

instance Monad Thread where
    return r = Pure r

    Yield next      >>= f = Yield (next >>= f)
    Fork left right >>= f = ...
~~~

# Monads

~~~ haskell
data Thread r
    = Yield (Thread r)
    | Fork  (Thread r) (Thread r)
    | Pure  r

instance Monad Thread where
    return r = Pure r

    Yield next      >>= f = Yield (next >>= f)
    Fork left right >>= f = Fork (left >>= f) (right >>= f)
    Pure r          >>= f = ...
~~~

# Monads

~~~ haskell
data Thread r
    = Yield (Thread r)
    | Fork  (Thread r) (Thread r)
    | Pure  r

instance Monad Thread where
    return r = Pure r

    Yield next      >>= f = Yield (next >>= f)
    Fork left right >>= f = Fork (left >>= f) (right >>= f)
    Pure r          >>= f = f r
~~~

# Monads

Again, Scala by default doesnt define `Monad`, instead classes like `List`,
`Map` and `Option` define their own `flatMap` methods. These `flatMap` methods
behave exactly like `(>>=)`.

Scala does provide some nice syntactic sugar for monadic operations, that is
very similar to do notation in Haskell.

~~~ scala
for {
    x <- List(1, 2)
    y <- List(3, 4)
    z = (x, y)
} yield z
-- List((1,3), (1,4), (2,3), (2,4))
~~~

This desugars to nested `flatMap`s

~~~ scala
List(1,2).flatMap(x => List(3,4).flatMap(y => List((x, y))))
~~~

This is a good example of the monadic effect of `List`, non-determinism! Think
of the `List` `Monad` as a computation that doesnt know what its result might
be, instead of operating on just one result, you operate on all possible
results.

Here is a interesting example of non-determinism in the `List` `Monad`.

~~~ haskell
filterM :: Monad m => (a -> m Bool) -> [a] -> m [a]
powerset :: [a] -> [[a]]
powerset xs = filterM (const [True, False]) xs
-- powerset [1,2,3] = [[1,2,3],[1,2],[1,3],[1],[2,3],[2],[3],[]]
~~~

# Monads

`Option` is a very useful abstraction that saves us from the dreaded
`NullPointerException`. With `Option`, we can tell the compiler "this
computation might fail to return a result", and the compiler will make sure that
we always handle the failure case. If we don't, the code won't compile!

Making use of `Option` can prevent many, many subtle bugs.

## 1
~~~ scala
for {
    x <- Some(4)
    y <- Nothing
    z = x + y
} yield z
~~~

## 2
~~~ scala
for {
    x <- Some(4)
    y <- Some(5)
    z = x + y
} yield z
~~~

# Monads

`Option` is a very useful abstraction that saves us from the dreaded
`NullPointerException`. With `Option`, we can tell the compiler "this
computation might fail to return a result", and the compiler will make sure that
we always handle the failure case. If we don't, the code won't compile!

Making use of `Option` can prevent many, many subtle bugs.

##1
~~~ scala
for {
    x <- Some(4)
    y <- Nothing
    z = x + y
} yield z
-- Nothing
~~~

##2
~~~ scala
for {
    x <- Some(4)
    y <- Some(5)
    z = x + y
} yield z
-- Some(9)
~~~

# Monads

The `Option` `Monad` represents a computation that could possibly fail to return
a result at one or more points. If any subcomputation returns `Nothing`, the
entire computation should return `Nothing`.

~~~ scala
def bothGrandFathers(person: Person): Option[(Person, Person)] =
    person.getFather() match {
        case Nothing => Nothing
        case Some(father) => father.getFather() match {
          case Nothing => Nothing
          case Some(grandFather1) => person.getMother() match {
            case Nothing => Nothing
            case Some(mother) => mother.getFather() match {
              case Nothing => Nothing
              case Some(grandFather1) =>
                Some((grandFather1, grandFather2))
            }
          }
        }
}
~~~

# Monads

A much more readable version using the `Option` `Monad`

~~~ scala
def bothGrandFathers(person: Person): Option[(Person, Person)] = for {
  father       <- person.getFather()
  grandFather1 <- father.getFather()
  mother       <- person.getMother()
  grandFather2 <- mother.getFather()
} yield (grandFather1, grandFather2)
~~~

# Monads

`Monads` are not some just some abstract Category Theory nonsense, they are
simply a common design pattern of sequencing effects that has been abstracted
into a simple interface.

Other `Monad`s can provide effects such as

* read-only/write-only state

* read/write state

* implicit arguments

* continuations

* parallelism/concurrency

* threadsafe, locally impure computations (ex. direct memory access) that appear pure to the outside world

* Software Transactional Memory

* I/O effects

* streaming resource handling

* Domain Specific Languages

# Monad Transformers

The real power of `Monad`s comes from their composability. You can choose which
side-effects you want to allow your computations access to
(alternatively, what side-effects you DON'T want them to have access to).

The compiler ensures you don't accidently use any side-effects that you
haven't explicitly mentioned, which helps reduce programming errors.

To layer `Monad`s on top of one another, we'll need `MonadTrans`

~~~ haskell
class MonadTrans (t :: (* -> *) -> * -> *) where
    lift :: Monad m => m a -> t m a
~~~

which satisfies these laws

~~~ haskell
lift (return x) == return x
lift (m >>= f)  == (lift m) >>= (\x -> lift (f x))
~~~

Whoa, that kind signature looks pretty scary!

~~~ haskell
t :: (* -> *) -- a Monad
  -> *        -- a concrete type
  -> *        -- we get a concrete type back
~~~

Unlike `Monad => Functor`, there is no guarantee that a valid `MonadTrans`
instance is a valid `Monad` instance from the typeclass definition alone.

# Monad Transformers

~~~ haskell
data Thread m r
    = Yield   (Thread m r)
    | Fork    (Thread m r) (Thread m r)
    | Lift (m (Thread m r))
    | Pure  r
    | Done
~~~

~~~ haskell
instance Monad m => Functor (Thread m) where
    fmap f (Yield next)      = Yield (fmap f next)
    fmap f (Fork left right) = Fork (fmap f left) (fmap f right)
    fmap f (Lift m)          = Lift (liftM (fmap f) m)
    fmap f (Pure r)          = Pure (f r)
    fmap f  Done             = Done
~~~

~~~ haskell
instance Monad m => Monad (Thread m) where
    return a = Pure a

    Yield next      >>= f = Yield (next >>= f)
    Fork left right >>= f = Fork (left >>= f) (right >>= f)
    Lift m          >>= f = Lift (m >>= f)
    Pure r          >>= f = f r
    Done            >>= f = Done
~~~

~~~ haskell
instance MonadTrans (Thread a) where
    lift m = ...
~~~

# Monad Transformers

~~~ haskell
data Thread m r
    = Yield   (Thread m r)
    | Fork    (Thread m r) (Thread m r)
    | Lift (m (Thread m r))
    | Pure  r
    | Done
~~~

~~~ haskell
instance Monad m => Functor (Thread m) where
    fmap f (Yield next)      = Yield (fmap f next)
    fmap f (Fork left right) = Fork (fmap f left) (fmap f right)
    fmap f (Lift m)          = Lift (liftM (fmap f) m)
    fmap f (Pure r)          = Pure (f r)
    fmap f  Done             = Done
~~~

~~~ haskell
instance Monad m => Monad (Thread m) where
    return a = Pure a

    Yield next      >>= f = Yield (next >>= f)
    Fork left right >>= f = Fork (left >>= f) (right >>= f)
    Lift m          >>= f = Lift (m >>= f)
    Pure r          >>= f = f r
    Done            >>= f = Done
~~~

~~~ haskell
instance MonadTrans (Thread a) where
    lift m = Lift (liftM Pure m)
~~~

# Monad Transformers

In the few lines of code in the last slide, we've created a fully usable
co-operative thread DSL. All we need are a few smart constructors.

~~~ haskell
yield :: Monad m => Thread m ()
yield = Yield (Pure ())

done :: Monad m => Thread m r
done = Done

fork :: Monad m => Thread m r -> Thread m ()
fork thread = do
    child <- Fork (Pure False) (Pure True)
    when child $ thread >> done
~~~

and a roundRobin thread scheduler

~~~ haskell
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
~~~

# Monad Transformers

~~~ haskell
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
~~~

Now if we run our `mainThread` using `roundRobin` we'll see

~~~
Forking thread #1
Forking thread #2
1
Hello World
2
Hello World
3
Hello World
4
5
6
7
8
9
10
~~~

We've used our `Thread` transformer to layer `IO` effects (printing to
stdout) with our threading effects.
