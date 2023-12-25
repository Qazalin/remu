
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_3_4>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_add_i32 s3, s14, -1                                      // 000000001714: 8103C10E
	s_mov_b32 s2, s15                                          // 000000001718: BE82000F
	s_cmp_lt_u32 s3, 2                                         // 00000000171C: BF0A8203
	s_cselect_b32 s3, -1, 0                                    // 000000001720: 980380C1
	s_cmp_eq_u32 s15, 1                                        // 000000001724: BF06810F
	s_cselect_b32 s8, -1, 0                                    // 000000001728: 980880C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000172C: BF870009
	s_and_b32 s3, s8, s3                                       // 000000001730: 8B030308
	s_mov_b32 s8, -1                                           // 000000001734: BE8800C1
	s_and_b32 vcc_lo, exec_lo, s3                              // 000000001738: 8B6A037E
	s_cbranch_vccnz 4                                          // 00000000173C: BFA40004 <E_3_4+0x50>
	s_ashr_i32 s15, s14, 31                                    // 000000001740: 860F9F0E
	s_mov_b32 s3, 0                                            // 000000001744: BE830080
	s_cbranch_execz 3                                          // 000000001748: BFA50003 <E_3_4+0x58>
	s_branch 10                                                // 00000000174C: BFA0000A <E_3_4+0x78>
	s_and_not1_b32 vcc_lo, exec_lo, s8                         // 000000001750: 916A087E
	s_cbranch_vccnz 8                                          // 000000001754: BFA40008 <E_3_4+0x78>
	s_mov_b32 s15, 0                                           // 000000001758: BE8F0080
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000175C: BF870009
	s_lshl_b64 s[8:9], s[14:15], 2                             // 000000001760: 8488820E
	s_waitcnt lgkmcnt(0)                                       // 000000001764: BF89FC07
	s_add_u32 s6, s6, s8                                       // 000000001768: 80060806
	s_addc_u32 s7, s7, s9                                      // 00000000176C: 82070907
	s_load_b32 s3, s[6:7], -0x4                                // 000000001770: F40000C3 F81FFFFC
	s_waitcnt lgkmcnt(0)                                       // 000000001778: BF89FC07
	s_load_b32 s6, s[0:1], null                                // 00000000177C: F4000180 F8000000
	s_lshl_b32 s0, s2, 2                                       // 000000001784: 84008202
	v_mov_b32_e32 v1, 0                                        // 000000001788: 7E020280
	s_ashr_i32 s1, s0, 31                                      // 00000000178C: 86019F00
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001790: BF870499
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001794: 84808200
	s_add_u32 s2, s4, s0                                       // 000000001798: 80020004
	s_waitcnt lgkmcnt(0)                                       // 00000000179C: BF89FC07
	v_mul_f32_e64 v0, s3, s6                                   // 0000000017A0: D5080000 00000C03
	s_addc_u32 s3, s5, s1                                      // 0000000017A8: 82030105
	s_lshl_b64 s[0:1], s[14:15], 2                             // 0000000017AC: 8480820E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017B0: BF870099
	s_add_u32 s0, s2, s0                                       // 0000000017B4: 80000002
	v_max_f32_e32 v0, 0, v0                                    // 0000000017B8: 20000080
	s_addc_u32 s1, s3, s1                                      // 0000000017BC: 82010103
	global_store_b32 v1, v0, s[0:1]                            // 0000000017C0: DC6A0000 00000001
	s_nop 0                                                    // 0000000017C8: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017CC: BFB60003
	s_endpgm                                                   // 0000000017D0: BFB00000
