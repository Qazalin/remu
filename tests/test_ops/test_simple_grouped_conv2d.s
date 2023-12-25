
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_2_2n1>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_lshl_b32 s8, s15, 1                                      // 000000001714: 8408810F
	s_mov_b32 s2, s15                                          // 000000001718: BE82000F
	s_ashr_i32 s9, s8, 31                                      // 00000000171C: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001720: BF870009
	s_lshl_b64 s[8:9], s[8:9], 2                               // 000000001724: 84888208
	s_waitcnt lgkmcnt(0)                                       // 000000001728: BF89FC07
	s_add_u32 s6, s6, s8                                       // 00000000172C: 80060806
	s_addc_u32 s7, s7, s9                                      // 000000001730: 82070907
	s_add_u32 s0, s0, s8                                       // 000000001734: 80000800
	s_addc_u32 s1, s1, s9                                      // 000000001738: 82010901
	s_load_b64 s[6:7], s[6:7], null                            // 00000000173C: F4040183 F8000000
	s_load_b64 s[0:1], s[0:1], null                            // 000000001744: F4040000 F8000000
	s_ashr_i32 s3, s15, 31                                     // 00000000174C: 86039F0F
	s_waitcnt lgkmcnt(0)                                       // 000000001750: BF89FC07
	v_fma_f32 v0, s6, s0, 0                                    // 000000001754: D6130000 02000006
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 00000000175C: BF8704B1
	v_fmac_f32_e64 v0, s7, s1                                  // 000000001760: D52B0000 00000207
	v_mov_b32_e32 v1, 0                                        // 000000001768: 7E020280
	s_lshl_b64 s[0:1], s[2:3], 2                               // 00000000176C: 84808202
	s_add_u32 s0, s4, s0                                       // 000000001770: 80000004
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001774: BF870002
	v_max_f32_e32 v0, 0, v0                                    // 000000001778: 20000080
	s_addc_u32 s1, s5, s1                                      // 00000000177C: 82010105
	global_store_b32 v1, v0, s[0:1]                            // 000000001780: DC6A0000 00000001
	s_nop 0                                                    // 000000001788: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 00000000178C: BFB60003
	s_endpgm                                                   // 000000001790: BFB00000
