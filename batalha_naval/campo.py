class cenario:
    
    map_size = 7
    nr_barcos = 3
    

    def __init__(self, new_nr_barcos=3):
        #   o status dos barcos será igual a quantidade de barcos iterados somados com o tamanho deles;
        #  e quando o valor for atingido, todos os barcos foram destruídos = fim de jogo
        self.st_barcos = 0

        # valor atual de posicoes de barcos atingidos
        self.vl_atualbarcos = 0
        self.campo   = [[]]
        for i in range(7):
            self.campo.append([])
            for j in range(7):
                self.campo[i].append('.')
        
        self.nr_barcos = new_nr_barcos
        for i in range(self.nr_barcos):
            self.st_barcos = self.st_barcos + i+1

    def exibe(self):
        print('  1 2 3 4 5 6 7')
        for i in range(self.map_size):
            linha = chr(65 + i)
            for j in range(self.map_size):
                linha = linha + ' ' + self.campo[i][j]
            
            print(linha)

    def set_navio(self,posX,posY,tam,dir='rigth'):
        try:
            px = ord(posX) - 65 #convertendo de letra para a respectiva linha
            py = int(posY) - 1

            nome = str(tam)

        
            if dir == 'rigth':
                for i in range(tam):
                    #validando se na posição já não existe uma navio
                    if self.campo[px][py+i] != '.':
                        return -1
                # setando o navio: necessário fazer loop separado porque caso haja interferencia
                # em outra linha que não a primeira, é inserido erroneamente uma parte do navio
                for i in range(tam):
                    self.campo[px][py+i] = nome
            else:
                for i in range(tam):
                    #validando se na posição já não existe uma navio
                    if self.campo[px+i][py] != '.':
                        return -1

                for i in range(tam):
                    self.campo[px+i][py] = nome

            return 0 #sem erros
        except:
            return -1 #erro de inserção, inserir nova posição/direção

    def retorna_tiro(self,posX,posY):
        try:
            px = ord(posX) - 65
            py = int(posY) - 1
        
            if self.campo[px][py] == '.':
                self.campo[px][py] = '*'
                return 0 #tiro na agua
            elif self.campo[px][py] == '*':
                return -3   #local já atingido
            else:
                if self.campo[px][py] == '@':
                    return -3 #local já atingido
                
                # se foi uma posição de barco ainda não atingido, itero a quantidade
                self.vl_atualbarcos = self.vl_atualbarcos + 1
                
                retorno = self.campo[px][py]
                self.campo[px][py] = '@'

                # valor atual de casas de barcos atingidos igual a quantidade
                # de casas de barcos = fim de jogo
                if self.vl_atualbarcos == self.st_barcos:
                    return -2

                return int(retorno) #tiro em parte do navio X
            
        except: #em caso de erro
            return -1