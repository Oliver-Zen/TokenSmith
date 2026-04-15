Topic: Concurrency control through conflict serializability and two-phase locking.

Question 1: What does it mean for two schedules to be conflict equivalent?

Correct textbook chunk(s): If a schedule S can be transformed into a schedule S′ by a series of swaps of non-conflicting instructions, we say that S and S′ are conflict equivalent

Question 2: When is a schedule conflict serializable?
Correct textbook chunk(s): We say that a schedule S is conflict serializable if it is conflict equivalent to a serial schedule.

Question 3: How does a precedence graph represent ordering constraints between transactions in a schedule?
Correct textbook chunk(s): Consider a schedule S. We construct a directed graph, called a precedence graph, from S. This graph consists of a pair G = (V, E), where V is a set of vertices and E is a set of edges. The set of vertices consists of all the transactions participating in the schedule. The set of edges consists of all edges Ti → Tj for which one of three conditions holds:
1.  Ti executes write(Q) before Tj executes read(Q).
2.  Ti executes read(Q) before Tj executes write(Q).
3.  Ti executes write(Q) before Tj executes write(Q).
If an edge Ti → Tj exists in the precedence graph, then, in any serial schedule S' equivalent to S, Ti must appear before Tj.

Question 4: What are the growing and shrinking phases of the two-phase locking protocol?
Correct textbook chunk(s): One protocol that ensures serializability is the two-phase locking protocol. This protocol requires that each transaction issue lock and unlock requests in two phases:
1. Growing phase. A transaction may obtain locks, but may not release any lock.
2. Shrinking phase. A transaction may release locks, but may not obtain any new locks.

Question 5: How does strict two-phase locking prevent cascading rollbacks?
Correct textbook chunk(s): Cascading rollbacks can be avoided by a modification of two-phase locking called the strict two-phase locking protocol. This protocol requires not only that locking be two phase, but also that all exclusive-mode locks taken by a transaction be held until that transaction commits. This requirement ensures that any data written by an uncommitted transaction are locked in exclusive mode until the transaction commits, preventing any other transaction from reading the data.






