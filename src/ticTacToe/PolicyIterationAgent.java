package ticTacToe;

import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * A policy iteration agent. You should implement the following methods:
 * (1) {@link PolicyIterationAgent#evaluatePolicy}: this is the policy evaluation step from your lectures
 * (2) {@link PolicyIterationAgent#improvePolicy}: this is the policy improvement step from your lectures
 * (3) {@link PolicyIterationAgent#train}: this is a method that should runs/alternate (1) and (2) until convergence. 
 * 
 * NOTE: there are two types of convergence involved in Policy Iteration: Convergence of the Values of the current policy, 
 * and Convergence of the current policy to the optimal policy.
 * The former happens when the values of the current policy no longer improve by much (i.e. the maximum improvement is less than 
 * some small delta). The latter happens when the policy improvement step no longer updates the policy, i.e. the current policy 
 * is already optimal. The algorithm should stop when this happens.
 * 
 * @author ae187
 *
 */
public class PolicyIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states according to the current policy (policy evaluation). 
	 */
	HashMap<Game, Double> policyValues=new HashMap<Game, Double>();
	
	/**
	 * This stores the current policy as a map from {@link Game}s to {@link Move}. 
	 */
	HashMap<Game, Move> curPolicy=new HashMap<Game, Move>();
	
	double discount=0.9;
	
	/**
	 * The mdp model used, see {@link TTTMDP}
	 */
	TTTMDP mdp;
	
	/**
	 * loads the policy from file if one exists. Policies should be stored in .pol files directly under the project folder.
	 */
	public PolicyIterationAgent() {
		super();
		this.mdp=new TTTMDP();
		initValues();
		initRandomPolicy();
		train();
		
		
	}
	
	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * @param p
	 */
	public PolicyIterationAgent(Policy p) {
		super(p);
		
	}

	/**
	 * Use this constructor to initialise a learning agent with default MDP paramters (rewards, transitions, etc) as specified in 
	 * {@link TTTMDP}
	 * @param discountFactor
	 */
	public PolicyIterationAgent(double discountFactor) {
		
		this.discount=discountFactor;
		this.mdp=new TTTMDP();
		initValues();
		initRandomPolicy();
		train();
	}

	/**
	 * Use this constructor to set the various parameters of the Tic-Tac-Toe MDP
	 * @param discountFactor
	 * @param winningReward
	 * @param losingReward
	 * @param livingReward
	 * @param drawReward
	 */
	public PolicyIterationAgent(double discountFactor, double winningReward, double losingReward, double livingReward, double drawReward)
	{
		this.discount=discountFactor;
		this.mdp=new TTTMDP(winningReward, losingReward, livingReward, drawReward);
		initValues();
		initRandomPolicy();
		train();
	}

	/**
	 * Initialises the {@link #policyValues} map, and sets the initial value of all states to 0 
	 * (V0 under some policy pi ({@link #curPolicy} from the lectures). Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do this. 
	 * 
	 */
	public void initValues()
	{
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
			this.policyValues.put(g, 0.0);
		
	}
	
	/**
	 *  You should implement this method to initially generate a random policy, i.e. fill the {@link #curPolicy} for every state. Take care that the moves you choose
	 *  for each state ARE VALID. You can use the {@link Game#getPossibleMoves()} method to get a list of valid moves and choose 
	 *  randomly between them. 
	 */
	public void initRandomPolicy()
	{
		Random random = new Random();
		List<Move> possibleMoves; //list containing all possible moves for a state 'game'

		//goes through all states (game) in the Game 
		for(Game game : this.policyValues.keySet()){

			//if game is a terminal state, skips it
			if( game.isTerminal() ){
				continue;}
		
			//stores all possible moves
			possibleMoves = game.getPossibleMoves(); 
			//generates a random index i within the possibleMoves bounds
			int i = random.nextInt(possibleMoves.size()); 
			//adds a random move to the currPolicy map storing initial policies
			this.curPolicy.put(game, possibleMoves.get(i));
		}
	}
	
	/**
	 * Performs policy evaluation steps until the maximum change in values is less than {@code delta}, in other words
	 * until the values under the currrent policy converge. After running this method, 
	 * the {@link PolicyIterationAgent#policyValues} map should contain the values of each reachable state under the current policy. 
	 * You should use the {@link TTTMDP} {@link PolicyIterationAgent#mdp} provided to do this.
	 *
	 * @param delta
	 */
	protected void evaluatePolicy(double delta)
	{	
		//initializing variables
		Move move;
		double newValue;
		double change;

		// runs the following section until convergence (change < delta)
		do {

			//creates a copy of the current policy values
			Map<Game, Double> policyValCopy = new HashMap<>(this.policyValues);
			change = 0;

			//goes over all states (games) 
			for( Game game : policyValCopy.keySet() ){

				//skips if a state is a terminal state
				if(game.isTerminal()){
					this.policyValues.put(game, 0.0);
					continue;
				}

				//gets the policy (move) for the given state
				move = this.curPolicy.get(game); 
				newValue = 0;
	
				//generates all transition probabilities and outcomes for the given game and move
				for( TransitionProb tp : mdp.generateTransitions(game, move) ){
	
					//gets all values needed for Q value computation
					double transitionProb = tp.prob;
					double livingReward = tp.outcome.localReward;
					double currentVal = this.policyValues.get(tp.outcome.sPrime);
					
					//computes the policy evaluation equation for the given policy
					newValue += transitionProb * (livingReward + discount * currentVal);
				}

				//updates the policy values with the new computed value
				this.policyValues.put(game, newValue);
				//tracks the maximum chanage in computed values and previous values for convergence 
				change = Math.max(change, Math.abs(newValue - policyValCopy.get(game)));
			}
		  //runs until the change in computed values is smaller than some small value delta (i.e. convergence)
		} while (change >= delta);
	}
		
	/**This method should be run AFTER the {@link PolicyIterationAgent#evaluatePolicy} train method to improve the current policy according to 
	 * {@link PolicyIterationAgent#policyValues}. You will need to do a single step of expectimax from each game (state) key in {@link PolicyIterationAgent#curPolicy} 
	 * to look for a move/action that potentially improves the current policy. 
	 * 
	 * @return true if the policy improved. Returns false if there was no improvement, i.e. the policy already returned the optimal actions.
	 */
	protected boolean improvePolicy()
	{
		double newValue;
		//creates a copy of the current policies
		Map<Game, Move> prevPolicyCopy = new HashMap<>(this.curPolicy);

		//goes over all states (games) 
		for( Game game : prevPolicyCopy.keySet() ){

			//single step expectimax

			//skips if a state is a terminal state
			if( game.isTerminal() ){
				this.policyValues.put(game, 0.0);
				continue;
			}
			
			//saves the current Q value of the policy
			double oldValue = this.policyValues.get(game);
	
			//goes through all possible moves for a game (state)
			for( Move move : game.getPossibleMoves() ){

				newValue = 0;
				
				//generates all transitions probabilities and outcomes for the given game and move
				for( TransitionProb tp : this.mdp.generateTransitions(game, move) ){

					//gets all values needed for Q value computation
					double transitionProb = tp.prob;
					double livingReward = tp.outcome.localReward;
					double currentVal = this.policyValues.get(tp.outcome.sPrime);

					//computes the value function
					newValue += transitionProb * (livingReward + discount * currentVal);
				}

				//updates current policy and policy values if the new value function is greater than 
				//the one computed for a fixed policy
				if( newValue > oldValue ){
					oldValue = newValue;
					this.curPolicy.put(game, move);
					this.policyValues.put(game, oldValue);
				}
			}
		}
		//returns true if any policy has been updated
		if( this.curPolicy.equals(prevPolicyCopy) ){
			return false;}
		else
			return true;
	}
	
	/**
	 * The (convergence) delta
	 */
	double delta=0.1;
	
	/**
	 * This method should perform policy evaluation and policy improvement steps until convergence (i.e. until the policy
	 * no longer changes), and so uses your 
	 * {@link PolicyIterationAgent#evaluatePolicy} and {@link PolicyIterationAgent#improvePolicy} methods.
	 */
	public void train()
	{
		//as long as policies are improved (improvePolicy function returns true),
		//it continues to first evaluate policies and then improve them untill there is no change in policy
		//(i.e. policy has convereged)
		do{
			this.evaluatePolicy(this.delta);
			
		} while( improvePolicy() );

		//a Policy object is created which holds all the updated current policies
		Policy newPolicy = new Policy(curPolicy);
		
		//assigns the policy object to the policy property of the super class
		super.policy = newPolicy;
	}
	
	public static void main(String[] args) throws IllegalMoveException
	{
		/**
		 * Test code to run the Policy Iteration Agent agains a Human Agent.
		 */
		PolicyIterationAgent pi=new PolicyIterationAgent();
		
		HumanAgent h=new HumanAgent();
		
		Game g=new Game(pi, h, h);
		
		g.playOut();

		//for testing the policy iteration agent against other agents for 50 times and printing the results in the terminal
		PolicyIterationAgent pl = new PolicyIterationAgent();
		
		Agent aa = new RandomAgent();
		int rw = 0;
		int rd = 0;
		int rl = 0;

		for( int i = 0; i < 50; i++ ) {
			Game gg = new Game(pl, aa, pl);
			gg.playOut();
			if( gg.state == Game.X_WON ) {
				rw++;
			} else if( gg.state == Game.O_WON ) {
				rl++;
			} else { rd++; }
		}
		
		aa = new AggressiveAgent();
		int aw = 0;
		int ad = 0;
		int al = 0;
		for( int i = 0; i < 50; i++ ){
			Game gg = new Game(pl, aa, pl);
			gg.playOut();
			if (gg.state == Game.X_WON) {
				aw++;
			} else if (gg.state == Game.O_WON) {
				al++;
			} else { ad++; }
		}
		
		aa = new DefensiveAgent();
		int dw = 0;
		int dd = 0;
		int dl = 0;
		for( int i = 0; i < 50; i++ ){
			Game gg = new Game(pl, aa, pl);
			gg.playOut();
			if (gg.state == Game.X_WON) {
				dw++;
			} else if (gg.state == Game.O_WON) {
				dl++;
			} else {
				dd++;
			}
		}

		System.err.println("Policy iteration agent playing against:");
		System.err.println("");
		System.err.println("-------------------------------");
		System.err.println("Against Random agent:");
		System.err.println("Wins: " + rw + ", Losses: " + rl + ", Draws: " + rd);
		System.err.println("");

		System.err.println("-------------------------------");
		System.err.println("Against Aggresive agent:");
		System.err.println("Wins: " + aw + ", Losses: " + al + ", Draws: " + ad);
		System.err.println("");

		System.err.println("-------------------------------");
		System.err.println("Against Defensive agent:");
		System.err.println("Wins: " + dw + ", Losses: " + dl + ", Draws: " + dd);
		System.err.println("");
		System.err.println("-------------------------------");

	}

}
