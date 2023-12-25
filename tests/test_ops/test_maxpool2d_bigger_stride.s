
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_3520_9_2_2>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001700: F4080000 F8000000
	s_mul_i32 s4, s15, 56                                      // 000000001708: 9604B80F
	s_mul_i32 s6, s14, 3                                       // 00000000170C: 9606830E
	s_ashr_i32 s5, s4, 31                                      // 000000001710: 86059F04
	v_dual_mov_b32 v0, 0xff800000 :: v_dual_mov_b32 v1, 0      // 000000001714: CA1000FF 00000080 FF800000
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000001720: 84848204
	s_waitcnt lgkmcnt(0)                                       // 000000001724: BF89FC07
	s_add_u32 s4, s2, s4                                       // 000000001728: 80040402
	s_addc_u32 s5, s3, s5                                      // 00000000172C: 82050503
	s_ashr_i32 s7, s6, 31                                      // 000000001730: 86079F06
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001734: BF870009
	s_lshl_b64 s[2:3], s[6:7], 2                               // 000000001738: 84828206
	s_mul_i32 s6, s15, 9                                       // 00000000173C: 9606890F
	s_add_u32 s2, s4, s2                                       // 000000001740: 80020204
	s_addc_u32 s3, s5, s3                                      // 000000001744: 82030305
	s_clause 0x1                                               // 000000001748: BF850001
	s_load_b64 s[4:5], s[2:3], null                            // 00000000174C: F4040101 F8000000
	s_load_b64 s[2:3], s[2:3], 0x70                            // 000000001754: F4040081 F8000070
	s_ashr_i32 s7, s6, 31                                      // 00000000175C: 86079F06
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001760: BF870499
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001764: 84868206
	s_add_u32 s6, s0, s6                                       // 000000001768: 80060600
	s_waitcnt lgkmcnt(0)                                       // 00000000176C: BF89FC07
	v_max3_f32 v0, s5, s4, v0                                  // 000000001770: D61C0000 04000805
	s_addc_u32 s4, s1, s7                                      // 000000001778: 82040701
	s_ashr_i32 s15, s14, 31                                    // 00000000177C: 860F9F0E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001780: BF870099
	s_lshl_b64 s[0:1], s[14:15], 2                             // 000000001784: 8480820E
	v_max3_f32 v0, s3, s2, v0                                  // 000000001788: D61C0000 04000403
	s_add_u32 s0, s6, s0                                       // 000000001790: 80000006
	s_addc_u32 s1, s4, s1                                      // 000000001794: 82010104
	global_store_b32 v1, v0, s[0:1]                            // 000000001798: DC6A0000 00000001
	s_nop 0                                                    // 0000000017A0: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017A4: BFB60003
	s_endpgm                                                   // 0000000017A8: BFB00000
