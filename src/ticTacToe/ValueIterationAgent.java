package ticTacToe;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A Value Iteration Agent, only very partially implemented. The methods to implement are: 
 * (1) {@link ValueIterationAgent#iterate}
 * (2) {@link ValueIterationAgent#extractPolicy}
 * 
 * You may also want/need to edit {@link ValueIterationAgent#train} - feel free to do this, but you probably won't need to.
 * @author ae187
 *
 */
public class ValueIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states
	 */
	Map<Game, Double> valueFunction=new HashMap<Game, Double>();
	
	/**
	 * the discount factor
	 */
	double discount=0.9;
	
	/**
	 * the MDP model
	 */
	TTTMDP mdp=new TTTMDP();
	
	/**
	 * the number of iterations to perform - feel free to change this/try out different numbers of iterations
	 */
	int k=10;
	
	/**
	 * This constructor trains the agent offline first and sets its policy
	 */
	public ValueIterationAgent()
	{
		super();
		mdp=new TTTMDP();
		this.discount=0.9;
		initValues();
		train();
	}
	
	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * @param p
	 */
	public ValueIterationAgent(Policy p) {
		super(p);
	}

	public ValueIterationAgent(double discountFactor) {
		
		this.discount=discountFactor;
		mdp=new TTTMDP();
		initValues();
		train();
	}
	
	/**
	 * Initialises the {@link ValueIterationAgent#valueFunction} map, and sets the initial value of all states to 0 
	 * (V0 from the lectures). Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do this. 
	 */
	public void initValues()
	{
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
			this.valueFunction.put(g, 0.0);
	}
	
	public ValueIterationAgent(double discountFactor, double winReward, double loseReward, double livingReward, double drawReward)
	{
		this.discount=discountFactor;
		mdp=new TTTMDP(winReward, loseReward, livingReward, drawReward);
	}
	
	 /** 
	 * Performs {@link #k} value iteration steps. After running this method, the {@link ValueIterationAgent#valueFunction} map should contain
	 * the (current) values of each reachable state. You should use the {@link TTTMDP} provided to do this.
	 */
	public void iterate()
	{		
		//does a value iteration for the tic tac toe MDP for k amount of times 
		for( int i = 0; i < k; i++ ){

			//stores a copy of the value functions for iteratoin
			Map<Game, Double> valueFunctionCopy = new HashMap<>(this.valueFunction);
			double newValue, maxValue;

			//goes through each state (game) in the MDP
			for( Game game : valueFunctionCopy.keySet() ){ 

				//performing expectimax

				//if game is a terminal state, skips the value iteration on that state
				if( game.isTerminal() ){
					this.valueFunction.put(game, 0.0);
					continue;
				}

				//holds a large negative value for performing expectimax (negative infinity) 
				maxValue = -Integer.MAX_VALUE;
				
				//goes through all possible moves (actions) in a given state game
				for( Move move : game.getPossibleMoves() ){

					newValue = 0; 
					//generates transition probabilities and outcomes for all possible moves (actions) in a game (state)
					for( TransitionProb tp : mdp.generateTransitions(game, move) ){

						//gets needed values to perform the bellman equation
						double transitionProb = tp.prob;
						double livingReward = tp.outcome.localReward;
						double currentVal = this.valueFunction.get(tp.outcome.sPrime);

						//holds the Q value of a given game and move
						newValue += transitionProb * (livingReward + discount * currentVal); 
					}
					//compares the Q values of different moves in a game and holds the maximum
					maxValue = Math.max(newValue, maxValue);
				}
				//each time the state (game) changes, updates the value function for that game
				this.valueFunction.put(game, maxValue); 
			}
		}
	}
	
	/** 
	 * This method should be run AFTER the train method to extract a policy according to {@link ValueIterationAgent#valueFunction}
	 * You will need to do a single step of expectimax from each game (state) key in {@link ValueIterationAgent#valueFunction} 
	 * to extract a policy.
	 * 
	 * @return the policy according to {@link ValueIterationAgent#valueFunction}
	 */
	public Policy extractPolicy()
	{
		//creates a new policy 
		Policy policy = new Policy();
		//the move which gives the maximum Q value
		Move bestMove = null;
		//variables used for iteration
		double maxValue, newValue;
		
		//goes through each state (game) in the MDP: starts expectimax
		for( Game game : this.valueFunction.keySet() ){

			//performing a single step expectimax

			//if game is a terminal state, skips the value iteration on that state
			if(game.isTerminal()){
				this.valueFunction.put(game, 0.0);
				continue;
			}

			maxValue = -Integer.MAX_VALUE;
			
			//goes through all possible moves (actions) in a given state game
			for( Move move : game.getPossibleMoves() ){
				newValue = 0; 

				//generates transition probabilities and outcomes for all possible moves (actions) in a game (state)
				for( TransitionProb tp : mdp.generateTransitions(game, move) ){

					//gets needed values to perform the bellman equation
					double transitionProb = tp.prob;
					double livingReward = tp.outcome.localReward;
					double currentVal = this.valueFunction.get(tp.outcome.sPrime);

					//holds the Q value of a given game and move
					newValue += transitionProb * (livingReward + discount * currentVal); 
				}

				//if the new Q value is greater than the max value, max value gets uodated and 
				//the corresponding move is saved as the selected policy
				if (newValue > maxValue){
					maxValue = newValue;
					bestMove = move;
				}
			}
			//selected policy is saved into the policy map for the MDP
			policy.policy.put(game, bestMove);
		}
		return policy;
	}
	
	/**
	 * This method solves the mdp using your implementation of {@link ValueIterationAgent#extractPolicy} and
	 * {@link ValueIterationAgent#iterate}. 
	 */
	public void train()
	{
		/**
		 * First run value iteration
		 */
		this.iterate();

		/**
		 * now extract policy from the values in {@link ValueIterationAgent#valueFunction} and set the agent's policy  
		 */
		super.policy=extractPolicy();
		
		if (this.policy==null)
		{
			System.out.println("Unimplemented methods! First implement the iterate() & extractPolicy() methods");
			//System.exit(1);
		}
	}

	public static void main(String a[]) throws IllegalMoveException
	{
		//Test method to play the agent against a human agent.
		ValueIterationAgent agent=new ValueIterationAgent();
		HumanAgent d=new HumanAgent();
		
		Game g=new Game(agent, d, d);
		g.playOut();

		//for testing the value iteration agent against other agents for 50 times and printing the results in the terminal
		ValueIterationAgent vl = new ValueIterationAgent();
		
		Agent aa = new RandomAgent();
		int rw = 0;
		int rd = 0;
		int rl = 0;

		for( int i = 0; i < 50; i++ ) {
			Game gg = new Game(vl, aa, vl);
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
			Game gg = new Game(vl, aa, vl);
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
			Game gg = new Game(vl, aa, vl);
			gg.playOut();
			if (gg.state == Game.X_WON) {
				dw++;
			} else if (gg.state == Game.O_WON) {
				dl++;
			} else {
				dd++;
			}
		}

		System.err.println("Value iteration agent playing against:");
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
