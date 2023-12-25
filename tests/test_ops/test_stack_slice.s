
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_3_4n1>:
	s_load_b128 s[4:7], s[0:1], null                           // 000000001700: F4080100 F8000000
	s_mov_b32 s2, s15                                          // 000000001708: BE82000F
	s_cmp_gt_i32 s15, 0                                        // 00000000170C: BF02800F
	s_mov_b32 s0, 0                                            // 000000001710: BE800080
	s_cbranch_scc0 6                                           // 000000001714: BFA10006 <E_3_4n1+0x30>
	s_cmp_eq_u32 s2, 1                                         // 000000001718: BF068102
	s_cbranch_scc1 14                                          // 00000000171C: BFA2000E <E_3_4n1+0x58>
	s_ashr_i32 s15, s14, 31                                    // 000000001720: 860F9F0E
	s_mov_b32 s1, 0                                            // 000000001724: BE810080
	s_cbranch_execz 11                                         // 000000001728: BFA5000B <E_3_4n1+0x58>
	s_branch 18                                                // 00000000172C: BFA00012 <E_3_4n1+0x78>
	s_ashr_i32 s15, s14, 31                                    // 000000001730: 860F9F0E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001734: BF870009
	s_lshl_b64 s[0:1], s[14:15], 2                             // 000000001738: 8480820E
	s_waitcnt lgkmcnt(0)                                       // 00000000173C: BF89FC07
	s_add_u32 s0, s6, s0                                       // 000000001740: 80000006
	s_addc_u32 s1, s7, s1                                      // 000000001744: 82010107
	s_load_b32 s0, s[0:1], null                                // 000000001748: F4000000 F8000000
	s_cmp_eq_u32 s2, 1                                         // 000000001750: BF068102
	s_cbranch_scc0 65522                                       // 000000001754: BFA1FFF2 <E_3_4n1+0x20>
	s_ashr_i32 s15, s14, 31                                    // 000000001758: 860F9F0E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000175C: BF870009
	s_lshl_b64 s[8:9], s[14:15], 2                             // 000000001760: 8488820E
	s_waitcnt lgkmcnt(0)                                       // 000000001764: BF89FC07
	s_add_u32 s8, s6, s8                                       // 000000001768: 80080806
	s_addc_u32 s9, s7, s9                                      // 00000000176C: 82090907
	s_load_b32 s1, s[8:9], null                                // 000000001770: F4000044 F8000000
	s_cmp_lt_i32 s2, 2                                         // 000000001778: BF048202
	s_mov_b32 s3, 0                                            // 00000000177C: BE830080
	s_cbranch_scc1 6                                           // 000000001780: BFA20006 <E_3_4n1+0x9c>
	s_lshl_b64 s[8:9], s[14:15], 2                             // 000000001784: 8488820E
	s_waitcnt lgkmcnt(0)                                       // 000000001788: BF89FC07
	s_add_u32 s6, s6, s8                                       // 00000000178C: 80060806
	s_addc_u32 s7, s7, s9                                      // 000000001790: 82070907
	s_load_b32 s3, s[6:7], null                                // 000000001794: F40000C3 F8000000
	s_waitcnt lgkmcnt(0)                                       // 00000000179C: BF89FC07
	v_add_f32_e64 v0, s0, s1                                   // 0000000017A0: D5030000 00000200
	s_lshl_b32 s0, s2, 2                                       // 0000000017A8: 84008202
	v_mov_b32_e32 v1, 0                                        // 0000000017AC: 7E020280
	s_ashr_i32 s1, s0, 31                                      // 0000000017B0: 86019F00
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 0000000017B4: BF8704D9
	s_lshl_b64 s[0:1], s[0:1], 2                               // 0000000017B8: 84808200
	v_add_f32_e32 v0, s3, v0                                   // 0000000017BC: 06000003
	s_add_u32 s2, s4, s0                                       // 0000000017C0: 80020004
	s_addc_u32 s3, s5, s1                                      // 0000000017C4: 82030105
	s_lshl_b64 s[0:1], s[14:15], 2                             // 0000000017C8: 8480820E
	s_add_u32 s0, s2, s0                                       // 0000000017CC: 80000002
	s_addc_u32 s1, s3, s1                                      // 0000000017D0: 82010103
	global_store_b32 v1, v0, s[0:1]                            // 0000000017D4: DC6A0000 00000001
	s_nop 0                                                    // 0000000017DC: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017E0: BFB60003
	s_endpgm                                                   // 0000000017E4: BFB00000
